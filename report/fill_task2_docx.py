"""Fill the Task 2 section of the supplied Word assignment template."""

from copy import deepcopy
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parent
SOURCE = ROOT / "Final Major Assignment Report 2026 Template.docx"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Final Major Assignment Report 2026.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
Q = lambda name: f"{{{W}}}{name}"
ET.register_namespace("w", W)


def text_of(element):
    return "".join(t.text or "" for t in element.iter(Q("t")))


def set_text(paragraph, value):
    ppr = paragraph.find(Q("pPr"))
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = ET.SubElement(paragraph, Q("r"))
    text = ET.SubElement(run, Q("t"))
    text.text = value


def new_paragraph(value, sample):
    p = ET.Element(Q("p"))
    ppr = sample.find(Q("pPr"))
    if ppr is not None:
        p.append(deepcopy(ppr))
    set_text(p, value)
    return p


def table(rows):
    tbl = ET.Element(Q("tbl"))
    props = ET.SubElement(tbl, Q("tblPr"))
    borders = ET.SubElement(props, Q("tblBorders"))
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = ET.SubElement(borders, Q(side))
        edge.set(Q("val"), "single")
        edge.set(Q("color"), "D9D9D9")
        edge.set(Q("sz"), "4")
    for row_index, row in enumerate(rows):
        tr = ET.SubElement(tbl, Q("tr"))
        tr_props = ET.SubElement(tr, Q("trPr"))
        ET.SubElement(tr_props, Q("cantSplit"))
        if row_index == 0:
            ET.SubElement(tr_props, Q("tblHeader"))
        for value in row:
            tc = ET.SubElement(tr, Q("tc"))
            p = ET.SubElement(tc, Q("p"))
            set_text(p, value)
    return tbl


with ZipFile(SOURCE) as zin:
    source_xml = zin.read("word/document.xml")
    document = ET.fromstring(source_xml)
    body = document.find(Q("body"))
    paragraphs = [p for p in body if p.tag == Q("p")]

    task2 = next(p for p in paragraphs if text_of(p).startswith("2. Task 2"))
    task3 = next(p for p in paragraphs if text_of(p).startswith("3. Task 3"))
    start, end = list(body).index(task2), list(body).index(task3)
    section = list(body)[start:end]
    sample = next(p for p in section if p.tag == Q("p") and text_of(p).startswith("Title:"))

    # Keep the supplied heading styles and replace only Task 2 body content.
    for item in section:
        if item.tag == Q("p") and item not in (task2,):
            if text_of(item).startswith("2.1.") or text_of(item).startswith("2.2.") or text_of(item).startswith("2.3."):
                continue
            body.remove(item)

    content = [
        ("2.1. Problem Description", [
            "Title: Five consecutive hourly bike rental forecasts using an encoder decoder network with attention.",
            "Research objective: Predict total rentals (cnt) for the next five consecutive hours from the previous 24 hours, and evaluate forecast quality at every horizon.",
            "Input: 24 hourly observations with 19 features, including historical cnt, calendar and weather variables, plus cyclic hour, month and weekday encodings. Output: five predicted cnt values for t+1 through t+5, where t is the last observed hour.",
            "Dataset and split: UCI Bike Sharing hour.csv contains 17,379 records from 2011–2012. Records are sorted chronologically and split 70%/15%/15%. The component counts casual and registered are excluded when predicting cnt. Scalers are fitted only on training data. Windows crossing a missing hour are removed, leaving 10,404 train, 2,603 validation and 2,469 test samples.",
            "EDA and preprocessing: The CSV has no blank cells or duplicate timestamps. There are 75 gaps covering 165 absent hours; windows crossing a gap are discarded. In training data, mean demand is highest at 17:00 (390.9) and lowest at 04:00 (5.2), supporting a 24-hour history and cyclic hour features. Mean counts differ across train, validation and test: 159.9, 281.0 and 235.7. Details are in report/EDA.md.",
            "Workflow: load and sort data; engineer cyclic features; split and scale causally; construct complete 24-to-5 windows; train the sequence model; select the best validation checkpoint; run autoregressive test inference; inverse transform and calculate metrics.",
        ]),
        ("2.2. Model Architecture and Design", [
            "A two-layer LSTM encoder with 64 hidden units per layer processes the 24 historical vectors. Its output at every hour forms the attention memory; its final hidden and cell states initialize a two-layer LSTM decoder.",
            "At each forecast step, Bahdanau additive attention scores the preceding decoder hidden state against all 24 encoder states. Softmax weights create a context vector. The decoder combines this context with the preceding target value; its readout predicts the next count. The first decoder input is the last observed cnt. In validation and testing, subsequent inputs are predictions, so no future target is revealed.",
            "The model has 138,497 trainable parameters. Training uses five-step MSE, AdamW (learning rate 0.001; weight decay 0.00001), batch size 128, dropout 0.2 and gradient clipping at norm 1. Teacher forcing starts at 0.5 and declines by epoch. A validation-based learning-rate scheduler and early stopping restore the best checkpoint. Seed 42 was used; the saved checkpoint was evaluated again on the test set to verify the reported scores.",
        ]),
        ("2.3. Experimental Results", [
            "For each forecast horizon, metrics compare all 2,469 test observations with predictions after inverse scaling. MAE = mean(|y − ŷ|); RMSE = sqrt(mean((y − ŷ)²)); R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²; NSE uses the same formula. Therefore standard R² and NSE are identical here. Macro average is the mean of horizon scores; overall pooled metrics use all 12,345 forecasted values together.",
            "Test results (MAE and RMSE are measured in rentals per hour):",
            table([
                ["Horizon", "R²", "NSE", "MAE", "RMSE"],
                ["t+1", "0.9408", "0.9408", "34.1007", "52.6394"],
                ["t+2", "0.9152", "0.9152", "40.7705", "62.8955"],
                ["t+3", "0.9052", "0.9052", "43.3751", "66.3678"],
                ["t+4", "0.8947", "0.8947", "45.9726", "69.9566"],
                ["t+5", "0.8781", "0.8781", "49.4454", "75.2998"],
                ["Macro", "0.9068", "0.9068", "42.7329", "65.4318"],
                ["Overall", "0.9069", "0.9069", "42.7329", "65.8717"],
            ]),
            "Accuracy falls with horizon: MAE rises from 34.10 to 49.45 between t+1 and t+5. The overall pooled R² and NSE are 0.9069. These results describe one chronological test split; without a no-attention ablation, they do not isolate the effect of attention. Overlapping forecast windows also mean pooled errors are not independent for statistical inference.",
            "On the same test windows, repeating the last count gives MAE 166.89 and RMSE 234.95; using the same hour of the previous day gives MAE 79.99 and RMSE 132.60. The model scores are lower, but these baselines do not isolate the attention component.",
            "Sources: Bahdanau, Cho and Bengio (2015), arXiv:1409.0473; Nash and Sutcliffe (1970), Journal of Hydrology 10(3), 282–290; scikit-learn documentation for r2_score, mean_absolute_error and root_mean_squared_error; UCI Bike Sharing dataset (dataset 275).",
        ]),
    ]

    # The three original subsection headings stay in place. Place content under each.
    for heading, entries in content:
        anchor = next(p for p in section if p.tag == Q("p") and text_of(p) == heading)
        at = list(body).index(anchor) + 1
        for entry in entries:
            node = entry if isinstance(entry, ET.Element) else new_paragraph(entry, sample)
            if isinstance(entry, str) and entry.startswith("Test results ("):
                ppr = node.find(Q("pPr"))
                if ppr is None:
                    ppr = ET.Element(Q("pPr"))
                    node.insert(0, ppr)
                ET.SubElement(ppr, Q("pageBreakBefore"))
            body.insert(at, node)
            at += 1

    xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    # ElementTree removes namespace declarations used only by mc:Ignorable.
    # Word requires those prefixes to remain declared even if no surviving
    # element uses them directly.
    source_root_start = source_xml.index(b"<w:document")
    source_root_end = source_xml.index(b">", source_root_start)
    source_namespaces = re.findall(rb'xmlns:([\w-]+)="([^"]+)"', source_xml[source_root_start:source_root_end])
    root_end = xml.index(b">", xml.index(b"<w:document"))
    root_tag = xml[:root_end]
    missing = b"".join(
        b' xmlns:' + prefix + b'="' + uri + b'"'
        for prefix, uri in source_namespaces
        if b'xmlns:' + prefix + b'=' not in root_tag
    )
    xml = xml[:root_end] + missing + xml[root_end:]
    with ZipFile(DEST, "w", ZIP_DEFLATED) as zout:
        for member in zin.infolist():
            data = xml if member.filename == "word/document.xml" else zin.read(member.filename)
            zout.writestr(member, data)

print(DEST)
