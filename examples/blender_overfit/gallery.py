"""Create a local review report from frozen references and available scene outputs."""

import argparse
import html
import json
from pathlib import Path


def build_gallery(dataset: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    manifest = json.loads((dataset / "manifest.json").read_text())
    summary_html = ""
    summary_path = dataset / "comparison-summary.json"
    if summary_path.is_file():
        comparison = json.loads(summary_path.read_text())
        summary = comparison["summary"]
        statistics_rows = []
        for name, key in [
            ("超限未交付计 0", "paired"),
            ("同步剔除超限未交付题", "paired_excluding_budget"),
        ]:
            statistics = summary[key]
            values = [
                "待完成" if statistics[field] is None else f"{statistics[field]:.4f}"
                for field in ["sft_mean", "agent_mean", "mean_agent_minus_sft"]
            ]
            statistics_rows.append(
                f"<tr><td>{name}</td><td>{statistics['count']}</td>"
                + "".join(f"<td>{value}</td>" for value in values)
                + "</tr>"
            )
        summary_html = (
            f"<p>部署模型：<code>{html.escape(summary['model'])}</code></p>"
            + (
                f"<p>部署权重：<code>{html.escape(summary['deployed_checkpoint'])}</code></p>"
                if summary.get("deployed_checkpoint")
                else ""
            )
            + "<p>下表仅汇总两组均已评分的成对题目。超限前已交付场景的，保留视觉评分；"
            "普通代码失败计 0，不单独触发成对剔除。</p>"
            "<table><tr><th>统计口径</th><th>成对题数</th><th>SFT均分</th>"
            "<th>Agent均分</th><th>Agent − SFT</th></tr>"
            + "".join(statistics_rows)
            + "</table>"
            + f'<p><a href="{html.escape(comparison["csv_path"])}">逐题 CSV</a> · '
            + f'<a href="{html.escape(comparison["json_path"])}">完整结果 JSON</a></p>'
        )
    rows = []
    sheet = Image.new("RGB", (1024, 4 * 288), "white")
    draw = ImageDraw.Draw(sheet)
    for index, task in enumerate(manifest["tasks"]):
        tiles = Image.new("RGB", (512, 512), "white")
        for image_index, reference in enumerate(task["reference_images"]):
            with Image.open(dataset / reference["path"]) as source:
                image = ImageOps.contain(source.convert("RGB"), (256, 256))
            x, y = (image_index % 2) * 256, (image_index // 2) * 256
            tiles.paste(
                image, (x + (256 - image.width) // 2, y + (256 - image.height) // 2)
            )
        preview = Path("tasks") / task["task_id"] / "references.jpg"
        tiles.save(dataset / preview, quality=90)
        thumb = tiles.resize((256, 256))
        x, y = (index % 4) * 256, (index // 4) * 288
        sheet.paste(thumb, (x, y))
        draw.text(
            (x + 4, y + 257),
            f"{index + 1:02d} source row {task['source_row_index']}",
            fill="black",
        )
        candidates = []
        for mode in ["sft", "agent"]:
            directory = Path("results") / mode / task["task_id"]
            path = directory / "preview.png"
            result = {}
            if (dataset / directory / "result.json").is_file():
                result = json.loads((dataset / directory / "result.json").read_text())
            score = result.get("score")
            caption = html.escape(str(result.get("status", "")))
            if score is not None:
                caption += f" · 视觉总分 {score:.4f}"
            if result.get("job_url"):
                caption += (
                    f' · <a href="{html.escape(result["job_url"])}">Arena任务</a>'
                )
            if result.get("evaluation_job_url"):
                caption += f' · <a href="{html.escape(result["evaluation_job_url"])}">评分任务</a>'
            if result.get("budget_reason"):
                caption += "<br>限制原因：" + html.escape(result["budget_reason"])
            if result.get("execution_error_type"):
                caption += "<br>执行错误：" + html.escape(
                    result["execution_error_type"]
                )
            if result.get("preview_kind") == "verifier_neutral":
                caption += "<br>此预览为评分用中性材质视图，原始场景文件未改动。"
            if (dataset / directory / "verification-preview.png").is_file():
                caption += (
                    f' · <a href="{directory}/verification-preview.png">评分视图</a>'
                )
                if result.get("preview_kind") != "verifier_neutral":
                    caption += (
                        "<details><summary>展开评分视图（中性材质）</summary>"
                        f'<img src="{directory}/verification-preview.png" '
                        'alt="固定评分取景"></details>'
                    )
            if (dataset / directory / "display/scene.blend").is_file():
                caption += f' · <a href="{directory}/display/scene.blend">展示版 scene.blend</a>'
            if (dataset / directory / "verification/render-manifest.json").is_file():
                caption += f' · <a href="{directory}/verification/render-manifest.json">评分证据清单</a>'
            for name, label in [
                ("generated.py", "生成代码"),
                ("blender.log", "执行日志"),
            ]:
                if (dataset / directory / name).is_file():
                    caption += f' · <a href="{directory}/{name}">{label}</a>'
            if (dataset / path).is_file():
                candidates.append(
                    f'<td><img src="{html.escape(str(path))}" alt="{mode}"><br>'
                    f'<a href="{directory}/scene.blend">scene.blend</a><p>{caption}</p></td>'
                )
            else:
                result_path = next(
                    (
                        dataset / directory / name
                        for name in [
                            "result.json",
                            "single-turn-result.json",
                            "model-failure.json",
                        ]
                        if (dataset / directory / name).is_file()
                    ),
                    None,
                )
                status = (
                    json.loads(result_path.read_text()).get("status", "UNKNOWN")
                    if result_path
                    else "尚未运行"
                )
                label = caption or html.escape(str(status))
                if (dataset / directory / "scene.blend").is_file():
                    label += f'<p><a href="{directory}/scene.blend">scene.blend</a></p>'
                elif result.get("budget_without_scene"):
                    label += (
                        "<p>超限且未交付：主口径计 0；剔除口径下两组同步移除此题。</p>"
                    )
                elif result.get("failure_kind") == "model_failure":
                    label += "<p>未成功交付场景，模型失败计 0 分。</p>"
                candidates.append("<td>" + label + "</td>")
        rows.append(
            f'<tr><td>{index + 1:02d}<br><img src="{preview}" alt="四张query参考图">'
            f"<p>{html.escape(task['description'])}</p>"
            f'<a href="{task["sft_messages"]}">完整SFT输入</a> · '
            f'<a href="{task["query"]}">任务query</a></td>'
            + "".join(candidates)
            + "</tr>"
        )
    sheet.save(dataset / "reference-contact-sheet.jpg", quality=92)
    document = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Blender overfit16</title>'
        "<style>body{font-family:system-ui;margin:24px;color:#222}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:12px;vertical-align:top}img{max-width:100%;height:auto}"
        "p{max-width:40em}th{background:#eee}</style><h1>Blender overfit16</h1>"
        "<p>同一批SFT已见题，seed=42。两种模式使用同一checkpoint。空白结果不代表模型失败。</p>"
        f"<p>Checkpoint: <code>{html.escape(manifest['checkpoint'])}</code></p>"
        "<p>视觉评分：0.4Q + 0.4D + 0.2A；几何项不适用。此页不包含训练目标代码。</p>"
        + summary_html
        + "<table><tr><th>Query参考图及描述</th><th>SFT单轮代码</th><th>Thinking + agentic</th></tr>"
        + "".join(rows)
        + "</table></html>"
    )
    (dataset / "review.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    build_gallery(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
