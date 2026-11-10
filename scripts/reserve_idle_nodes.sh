#!/usr/bin/env bash
# Auto-discover free Slurm nodes and create/update/delete a reservation.
#
# A node counts as "free" when its sinfo state is:
#   - idle: unreserved and free; or
#   - reserved: idle inside a reservation, but only when it already belongs to
#     reservation NAME (free nodes held by other reservations are never taken).
#
# Usage:
#   reserve_idle_nodes.sh create [-n NAME] [-u USER] [-d DURATION] [-p PARTITION] [-m MIN] [-c COUNT] [-x EXCLUDE]
#   reserve_idle_nodes.sh update [-n NAME] [-u USER] [-d DURATION] [-p PARTITION] [-m MIN] [-c COUNT] [-x EXCLUDE] [-y]
#   reserve_idle_nodes.sh append [-n NAME] [-p PARTITION] [-x EXCLUDE] -c COUNT
#   reserve_idle_nodes.sh delete [-n NAME]
#   reserve_idle_nodes.sh show   [-n NAME]
#   reserve_idle_nodes.sh jobs   [-n NAME] [-v]
#   reserve_idle_nodes.sh jobs-all [-v]
#   reserve_idle_nodes.sh list-idle [-n NAME] [-p PARTITION] [-x EXCLUDE]
#
# NOTE: "update" REPLACES the reservation's node list with the discovered free
# set. It shows the nodes that would drop out of the reservation and asks for
# confirmation; pass -y to skip the prompt (required when non-interactive).
#
# Examples:
#   reserve_idle_nodes.sh create -n swe -u admin
#   reserve_idle_nodes.sh update -n swe -x slurmd-[1-5]    # replace nodes with up to -c free
#   reserve_idle_nodes.sh append -n swe -c 6               # add 6 more free nodes to existing
#   reserve_idle_nodes.sh delete -n swe
#   reserve_idle_nodes.sh jobs   -n swe                    # list jobs on reservation nodes
#   reserve_idle_nodes.sh jobs   -n swe -v                 # also show WorkDir/Command per job
#   reserve_idle_nodes.sh jobs-all                         # reservations + unreserved nodes + pending queue
#   reserve_idle_nodes.sh jobs-all -v                      # ... plus WorkDir/Command per job

set -euo pipefail

NAME="swe"
USER_NAME="${USER:-admin}"
DURATION="UNLIMITED"
PARTITION=""
MIN_NODES=1
COUNT=0   # 0 means: take all idle nodes; >0 means: cap to this many
EXCLUDE=""
FLAGS="OVERLAP,IGNORE_JOBS"
VERBOSE=0
FORCE_YES=0

ACTION="${1:-}"
shift || true

while getopts ":n:u:d:p:m:c:x:f:vy" opt; do
  case $opt in
    n) NAME="$OPTARG" ;;
    u) USER_NAME="$OPTARG" ;;
    d) DURATION="$OPTARG" ;;
    p) PARTITION="$OPTARG" ;;
    m) MIN_NODES="$OPTARG" ;;
    c) COUNT="$OPTARG" ;;
    x) EXCLUDE="$OPTARG" ;;
    f) FLAGS="$OPTARG" ;;
    v) VERBOSE=1 ;;
    y) FORCE_YES=1 ;;
    *) echo "Unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require sinfo
require scontrol
require squeue

own_reservation_nodes() {
  # Expand reservation $NAME's current node list, one per line (lex-sorted).
  # Prints nothing if the reservation does not exist.
  local nodes
  nodes=$(scontrol show reservation "$NAME" -o 2>/dev/null \
          | grep -oE 'Nodes=[^ ]+' | head -n1 | cut -d= -f2- || true)
  if [[ -z "$nodes" || "$nodes" == "(null)" ]]; then
    return 0
  fi
  scontrol show hostnames "$nodes" 2>/dev/null | sort -u || true
}

discover_idle_nodes() {
  # Print one free node name per line. Free nodes inside a reservation report
  # state "reserved" (not "idle"), so accept both — but only take
  # reserved-state nodes that already belong to $NAME, never another
  # reservation's. States with a "*" suffix (not responding) are skipped.
  local args=(-h -N -o "%N %T")
  [[ -n "$PARTITION" ]] && args+=(-p "$PARTITION")

  local state_lines idle_free resv_free
  state_lines=$(sinfo "${args[@]}")
  idle_free=$(awk '$2=="idle"{print $1}' <<<"$state_lines" | sort -u)
  resv_free=$(awk '$2=="reserved"{print $1}' <<<"$state_lines" | sort -u)

  if [[ -n "$resv_free" ]]; then
    local own
    own=$(own_reservation_nodes)
    if [[ -n "$own" ]]; then
      resv_free=$(comm -12 <(printf '%s\n' "$resv_free") <(printf '%s\n' "$own"))
    else
      resv_free=""
    fi
  fi

  local raw
  raw=$(printf '%s\n%s\n' "$idle_free" "$resv_free" | sed '/^$/d' | sort -u)
  if [[ -z "$raw" ]]; then
    return 0
  fi

  if [[ -n "$EXCLUDE" ]]; then
    # Expand exclude hostlist and filter
    local excl
    excl=$(scontrol show hostnames "$EXCLUDE" | sort -u)
    comm -23 <(printf '%s\n' "$raw") <(printf '%s\n' "$excl")
  else
    printf '%s\n' "$raw"
  fi
}

fold_nodes() {
  # Fold a newline-separated node list into a Slurm hostlist expression.
  local nodes
  nodes=$(cat | paste -sd, -)
  [[ -z "$nodes" ]] && return 1
  scontrol show hostlistsorted "$nodes"
}

reservation_exists() {
  scontrol show reservation "$NAME" >/dev/null 2>&1
}

do_show() {
  if reservation_exists; then
    scontrol show reservation "$NAME"
  else
    echo "Reservation '$NAME' does not exist."
    exit 1
  fi
}

do_list_idle() {
  local nodes
  nodes=$(discover_idle_nodes)
  if [[ -z "$nodes" ]]; then
    echo "No free nodes found (state idle, or reserved-free members of '$NAME')."
    return 0
  fi
  local n
  n=$(printf '%s\n' "$nodes" | wc -l)
  echo "Free nodes: $n (state idle, or reserved-free members of '$NAME')"
  echo "$nodes" | fold_nodes
}

confirm_node_drop() {
  # "update" replaces the node list wholesale: warn about nodes that would be
  # dropped from reservation $NAME and require confirmation (skip with -y).
  local new_list="$1"
  local current dropped
  current=$(own_reservation_nodes)
  if [[ -z "$current" ]]; then
    return 0
  fi
  dropped=$(comm -23 <(printf '%s\n' "$current") \
                     <(printf '%s\n' "$new_list" | sort -u))
  if [[ -z "$dropped" ]]; then
    return 0
  fi

  local dropped_hostlist
  dropped_hostlist=$(printf '%s\n' "$dropped" | fold_nodes)
  echo "WARNING: update replaces the node list; dropping from '$NAME': $dropped_hostlist"
  if (( FORCE_YES == 1 )); then
    return 0
  fi
  if [[ -t 0 ]]; then
    local ans
    read -r -p "Proceed and drop these nodes? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
      echo "Aborted." >&2
      exit 1
    fi
  else
    echo "Refusing to drop nodes without confirmation (non-interactive). Re-run with -y." >&2
    exit 1
  fi
}

do_create_or_update() {
  local mode="$1"  # create | update
  local idle_list
  idle_list=$(discover_idle_nodes)
  local total=0
  [[ -n "$idle_list" ]] && total=$(printf '%s\n' "$idle_list" | wc -l)

  if (( total < MIN_NODES )); then
    echo "Found $total free nodes; need at least $MIN_NODES. Aborting." >&2
    exit 1
  fi

  if (( COUNT > 0 )); then
    if (( COUNT > total )); then
      echo "Requested $COUNT nodes but only $total free. Aborting." >&2
      exit 1
    fi
    idle_list=$(printf '%s\n' "$idle_list" | head -n "$COUNT")
  fi
  local count
  count=$(printf '%s\n' "$idle_list" | wc -l)

  local hostlist
  hostlist=$(printf '%s\n' "$idle_list" | fold_nodes)
  echo "Discovered $count free nodes: $hostlist"

  if [[ "$mode" == "create" ]]; then
    if reservation_exists; then
      echo "Reservation '$NAME' already exists. Use 'update' instead." >&2
      exit 1
    fi
    set -x
    sudo scontrol create reservation \
      ReservationName="$NAME" \
      StartTime=now \
      Duration="$DURATION" \
      Users="$USER_NAME" \
      Nodes="$hostlist" \
      Flags="$FLAGS"
    set +x
  else
    if ! reservation_exists; then
      echo "Reservation '$NAME' does not exist. Creating instead."
    fi
    if reservation_exists; then
      confirm_node_drop "$idle_list"
      set -x
      sudo scontrol update ReservationName="$NAME" Nodes="$hostlist"
      set +x
    else
      set -x
      sudo scontrol create reservation \
        ReservationName="$NAME" \
        StartTime=now \
        Duration="$DURATION" \
        Users="$USER_NAME" \
        Nodes="$hostlist" \
        Flags="$FLAGS"
      set +x
    fi
  fi

  scontrol show reservation "$NAME"
}

do_append() {
  if (( COUNT <= 0 )); then
    echo "append requires -c COUNT (>0): how many extra idle nodes to add." >&2
    exit 2
  fi
  if ! reservation_exists; then
    echo "Reservation '$NAME' does not exist. Use 'create' first." >&2
    exit 1
  fi

  local current
  current=$(scontrol show reservation "$NAME" | awk -F'[ =]' '/Nodes=/{for(i=1;i<=NF;i++) if($i=="Nodes"){print $(i+1); exit}}')
  if [[ -z "$current" ]]; then
    echo "Could not parse current Nodes from reservation '$NAME'." >&2
    exit 1
  fi

  local idle_list
  idle_list=$(discover_idle_nodes)
  # Nodes the reservation already holds (its own reserved-free members) are
  # no-ops for append; only nodes outside the current list are candidates.
  if [[ -n "$idle_list" ]]; then
    idle_list=$(comm -23 <(printf '%s\n' "$idle_list") \
                         <(scontrol show hostnames "$current" | sort -u))
  fi
  local total=0
  [[ -n "$idle_list" ]] && total=$(printf '%s\n' "$idle_list" | wc -l)
  if (( total < COUNT )); then
    echo "Requested $COUNT extra nodes but only $total free candidates. Aborting." >&2
    exit 1
  fi

  local extra
  extra=$(printf '%s\n' "$idle_list" | head -n "$COUNT")
  local extra_list
  extra_list=$(printf '%s\n' "$extra" | fold_nodes)

  local merged
  merged=$(scontrol show hostlistsorted "${current},${extra_list}")
  echo "Current nodes:  $current"
  echo "Adding $COUNT free: $extra_list"
  echo "Merged nodes:   $merged"

  set -x
  sudo scontrol update ReservationName="$NAME" Nodes="$merged"
  set +x
  scontrol show reservation "$NAME"
}

do_delete() {
  if ! reservation_exists; then
    echo "Reservation '$NAME' does not exist; nothing to delete."
    exit 0
  fi
  set -x
  sudo scontrol delete ReservationName="$NAME"
  set +x
}

do_jobs() {
  if ! reservation_exists; then
    echo "Reservation '$NAME' does not exist." >&2
    exit 1
  fi
  print_resv_jobs "$NAME"
}

do_jobs_all() {
  local reservation_output names
  if ! reservation_output=$(scontrol show reservation -o 2>/dev/null); then
    echo "Failed to query reservations." >&2
    return 1
  fi
  names=$(grep -oE 'ReservationName=[^ ]+' <<<"$reservation_output" \
          | cut -d= -f2- || true)
  local count=0
  [[ -n "$names" ]] && count=$(printf '%s\n' "$names" | grep -c .)
  echo "Found $count reservation(s) (plus an unreserved section below)."
  echo
  local first=1
  if [[ -n "$names" ]]; then
    while IFS= read -r r; do
      [[ -z "$r" ]] && continue
      if (( first == 0 )); then echo; fi
      echo "============================================================"
      print_resv_jobs "$r"
      first=0
    done <<< "$names"
  fi
  # Also report the nodes that belong to no reservation at all.
  if (( first == 0 )); then echo; fi
  echo "============================================================"
  print_unreserved_jobs
  echo
  echo "============================================================"
  print_queued_jobs
}

print_resv_jobs() {
  local resv="$1"

  # Parse the Nodes= field from `scontrol show reservation -o` (one-line form).
  local nodes
  nodes=$(scontrol show reservation "$resv" -o 2>/dev/null \
          | grep -oE 'Nodes=[^ ]+' | head -n1 | cut -d= -f2-)
  if [[ -z "$nodes" ]]; then
    echo "Reservation : $resv"
    echo "  (could not parse Nodes=)"
    return 0
  fi

  local resv_expanded
  resv_expanded=$(scontrol show hostnames "$nodes" 2>/dev/null | sort -u)
  print_nodes_report "Reservation : $resv" "$nodes" "$resv_expanded"
}

print_unreserved_jobs() {
  # Report the nodes that belong to NO reservation at all: every node Slurm
  # knows about, minus the union of all reservations' node lists.
  local all_known resv_all nodes_display
  all_known=$(sinfo -h -N -o "%N" 2>/dev/null | sort -u)

  # Union of every reservation's expanded node list.
  local resv_nodes_fields
  resv_nodes_fields=$(scontrol show reservation -o 2>/dev/null \
                      | grep -oE 'Nodes=[^ ]+' | cut -d= -f2- | grep -v '^(null)$' || true)
  resv_all=""
  if [[ -n "$resv_nodes_fields" ]]; then
    resv_all=$(while IFS= read -r nl; do
                 [[ -z "$nl" ]] && continue
                 scontrol show hostnames "$nl" 2>/dev/null || true
               done <<< "$resv_nodes_fields" | sort -u)
  fi

  local unreserved
  if [[ -n "$resv_all" ]]; then
    unreserved=$(comm -23 <(printf '%s\n' "$all_known") <(printf '%s\n' "$resv_all"))
  else
    unreserved="$all_known"
  fi

  if [[ -n "$unreserved" ]]; then
    nodes_display=$(printf '%s\n' "$unreserved" | fold_nodes)
  else
    nodes_display="(none)"
  fi
  print_nodes_report "Section     : (unreserved — in no reservation)" \
                     "$nodes_display" "$unreserved"
}

job_type_from_gres() {
  # squeue %b reports requested generic resources, for example:
  #   N/A, gres/gpu:0, gres/gpu:8, or gres/gpu:a100:8
  # Treat an explicit zero-GPU request as CPU-only. This classifies the
  # resource request, not whether the process is actively using its GPUs.
  local gres="$1"
  local gpu_gres_re='(^|,)(gres/)?gpu(:[^,:[:space:]]+)?:[1-9][0-9]*([,(]|$)'
  if [[ "$gres" =~ $gpu_gres_re ]]; then
    echo "GPU"
  else
    echo "CPU"
  fi
}

format_pipe_table() {
  awk -F'|' '
    {
      row[NR] = $0
      if (NF > columns) columns = NF
      for (i = 1; i <= NF; i++) {
        if (length($i) > width[i]) width[i] = length($i)
      }
    }
    END {
      for (r = 1; r <= NR; r++) {
        split(row[r], cell, FS)
        for (i = 1; i < columns; i++) printf "%-*s  ", width[i], cell[i]
        print cell[columns]
      }
    }
  '
}

print_job_table() {
  local hostlist="$1"
  local rows
  rows=$(squeue -h --states=RUNNING -w "$hostlist" \
         -o "%i|%P|%j|%u|%t|%M|%D|%b|%R" 2>/dev/null || true)

  # Keep variable-length fields at the end, then calculate widths from the
  # actual result set so long job names do not shift TYPE/GRES out of column.
  {
    printf '%s\n' \
      "JOBID|PARTITION|USER|ST|TIME|NODES|TYPE|GRES|NODELIST(REASON)|NAME"
    if [[ -n "$rows" ]]; then
      local jobid partition jobname user state time nnodes gres reason job_type
      while IFS='|' read -r jobid partition jobname user state time nnodes gres reason; do
        job_type=$(job_type_from_gres "$gres")
        printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
          "$jobid" "$partition" "$user" "$state" "$time" "$nnodes" \
          "$job_type" "$gres" "$reason" "$jobname"
      done <<< "$rows"
    fi
  } | format_pipe_table
}

print_queued_jobs() {
  local rows
  if ! rows=$(squeue -h --states=PENDING \
              -o "%i|%P|%j|%u|%t|%M|%D|%b|%v|%R"); then
    echo "Failed to query pending jobs." >&2
    return 1
  fi

  echo "Queued jobs (PENDING):"
  if [[ -z "$rows" ]]; then
    echo "  (no pending jobs)"
    return 0
  fi

  {
    printf '%s\n' \
      "JOBID|PARTITION|USER|ST|TIME|NODES|TYPE|GRES|RESERVATION|REASON|NAME"
    local jobid partition jobname user state time nnodes gres reservation reason job_type
    while IFS='|' read -r jobid partition jobname user state time nnodes gres reservation reason; do
      job_type=$(job_type_from_gres "$gres")
      [[ "$reservation" == "(null)" ]] && reservation="(none)"
      printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$jobid" "$partition" "$user" "$state" "$time" "$nnodes" \
        "$job_type" "$gres" "$reservation" "$reason" "$jobname"
    done <<< "$rows"
  } | format_pipe_table
}

print_nodes_report() {
  # Shared reporting core for a set of nodes.
  #   $1 title       — first header line, already formatted (e.g. "Reservation : swe")
  #   $2 nodes_display — the hostlist string to print on the "Nodes :" line
  #   $3 expanded    — newline-separated node names to analyze (may contain stale)
  local title="$1" nodes_display="$2" expanded="$3"

  # Intersect with nodes Slurm currently knows about. A reservation can outlive
  # a node (e.g. removed from slurm.conf), and `squeue -w` / `sinfo -n` abort on
  # the *whole* list if any single name is unknown — so strip stale entries.
  local all_known valid stale
  all_known=$(sinfo -h -N -o "%N" 2>/dev/null | sort -u)
  valid=$(comm -12 <(printf '%s\n' "$expanded") <(printf '%s\n' "$all_known"))
  stale=$(comm -23 <(printf '%s\n' "$expanded") <(printf '%s\n' "$all_known"))

  local total stale_count valid_hostlist
  total=$(printf '%s\n' "$expanded" | grep -c .)
  stale_count=$(printf '%s\n' "$stale" | grep -c . || true)
  if [[ -n "$valid" ]]; then
    valid_hostlist=$(printf '%s\n' "$valid" | fold_nodes)
  else
    valid_hostlist=""
  fi

  local alloc=0 mix=0 free=0 unavail=0 other=0
  local free_hostlist="" unavail_hostlist=""
  if [[ -n "$valid_hostlist" ]]; then
    local states
    states=$(sinfo -h -N -n "$valid_hostlist" -o "%N %T" 2>/dev/null | sort -u || true)
    # Free capacity inside a reservation shows state "reserved", not "idle".
    # A "*" suffix means not responding — bucket those under unavail.
    alloc=$(echo "$states" | awk '$2=="allocated"{c++} END{print c+0}')
    mix=$(  echo "$states" | awk '$2=="mixed"{c++}     END{print c+0}')
    free=$( echo "$states" | awk '$2=="idle" || $2=="reserved" {c++} END{print c+0}')
    unavail=$(echo "$states" | awk '$2 ~ /drain|down|fail|maint|unk|\*/ {c++} END{print c+0}')
    other=$(echo "$states" | awk 'NF>0 && $2!="allocated" && $2!="mixed" && $2!="idle" && $2!="reserved" && $2 !~ /drain|down|fail|maint|unk|\*/ {c++} END{print c+0}')
    if (( free > 0 )); then
      free_hostlist=$(echo "$states" | awk '$2=="idle" || $2=="reserved" {print $1}' | fold_nodes)
    fi
    if (( unavail > 0 )); then
      unavail_hostlist=$(echo "$states" | awk '$2 ~ /drain|down|fail|maint|unk|\*/ {print $1}' | fold_nodes)
    fi
  fi

  echo "$title"
  echo "Nodes       : $nodes_display"
  echo "Summary     : total=$total  allocated=$alloc  mixed=$mix  free=$free  unavail=$unavail  other=$other"
  echo "Free        : ${free_hostlist:-(none)}"
  if (( unavail > 0 )); then
    echo "Unavail     : $unavail_hostlist"
  fi
  if (( stale_count > 0 )); then
    local stale_hostlist
    stale_hostlist=$(printf '%s\n' "$stale" | fold_nodes)
    echo "Stale       : $stale_count node(s) not known to Slurm (skipped): $stale_hostlist"
  fi
  echo

  echo "Jobs running on these nodes:"
  if [[ -z "$valid_hostlist" ]]; then
    echo "  (no valid nodes to query)"
  else
    print_job_table "$valid_hostlist"
  fi

  if (( VERBOSE == 1 )); then
    echo
    echo "Job details (WorkDir / Command / StdOut):"
    local jids=""
    [[ -n "$valid_hostlist" ]] && jids=$(squeue -h --states=RUNNING -w "$valid_hostlist" -o "%i" 2>/dev/null || true)
    if [[ -z "$jids" ]]; then
      echo "  (no running jobs)"
      return 0
    fi
    while IFS= read -r jid; do
      [[ -z "$jid" ]] && continue
      local info jobname workdir command stdout user
      info=$(scontrol show job "$jid" -o 2>/dev/null || true)
      [[ -z "$info" ]] && continue
      jobname=$(echo "$info" | grep -oE 'JobName=[^ ]+' | head -n1 | cut -d= -f2-)
      user=$(   echo "$info" | grep -oE 'UserId=[^ ]+'  | head -n1 | cut -d= -f2-)
      workdir=$(echo "$info" | grep -oE 'WorkDir=[^ ]+' | head -n1 | cut -d= -f2-)
      command=$(echo "$info" | grep -oE 'Command=[^ ]+' | head -n1 | cut -d= -f2-)
      stdout=$( echo "$info" | grep -oE 'StdOut=[^ ]+'  | head -n1 | cut -d= -f2-)
      echo "  JobId=$jid  User=$user  Name=$jobname"
      [[ -n "$workdir" ]] && echo "    WorkDir : $workdir"
      [[ -n "$command" ]] && echo "    Command : $command"
      [[ -n "$stdout" ]]  && echo "    StdOut  : $stdout"
    done <<< "$jids"
  fi
}

case "$ACTION" in
  create)    do_create_or_update create ;;
  update)    do_create_or_update update ;;
  append)    do_append ;;
  delete)    do_delete ;;
  show)      do_show ;;
  jobs)      do_jobs ;;
  jobs-all)  do_jobs_all ;;
  list-idle) do_list_idle ;;
  ""|-h|--help|help)
    sed -n '2,31p' "$0"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    sed -n '2,31p' "$0"
    exit 2
    ;;
esac
