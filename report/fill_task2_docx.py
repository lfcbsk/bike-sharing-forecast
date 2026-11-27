"""Insert the seven-section Task 2 report into the supplied Word template."""

from pathlib import Path
import re
import struct
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parent
SOURCE = ROOT / "Final Major Assignment Report 2026 Template.docx"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Final Major Assignment Report 2026 revised.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_R = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
q = lambda ns, name: f"{{{ns}}}{name}"
for prefix, uri in (("w", W), ("a", A), ("wp", WP), ("pic", PIC), ("r", R)):
    ET.register_namespace(prefix, uri)


def text_of(element):
    return "".join(t.text or "" for t in element.iter(q(W, "t")))


def paragraph(value, kind="body", keep_next=False, page_break=False):
    p = ET.Element(q(W, "p"))
    ppr = ET.SubElement(p, q(W, "pPr"))
    spacing = ET.SubElement(ppr, q(W, "spacing"))
    spacing.set(q(W, "after"), "100" if kind == "body" else "150")
    spacing.set(q(W, "line"), "276")
    if keep_next:
        ET.SubElement(ppr, q(W, "keepNext"))
    if page_break:
        ET.SubElement(ppr, q(W, "pageBreakBefore"))
    run = ET.SubElement(p, q(W, "r"))
    rpr = ET.SubElement(run, q(W, "rPr"))
    if kind != "body":
        ET.SubElement(rpr, q(W, "b"))
    size = ET.SubElement(rpr, q(W, "sz"))
    size.set(q(W, "val"), {"h2": "26", "h3": "22", "caption": "20"}.get(kind, "22"))
    t = ET.SubElement(run, q(W, "t"))
    t.text = value
    return p


def table(rows):
    tbl = ET.Element(q(W, "tbl"))
    props = ET.SubElement(tbl, q(W, "tblPr"))
    borders = ET.SubElement(props, q(W, "tblBorders"))
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = ET.SubElement(borders, q(W, side))
        edge.set(q(W, "val"), "single")
        edge.set(q(W, "color"), "D9D9D9")
        edge.set(q(W, "sz"), "4")
    grid = ET.SubElement(tbl, q(W, "tblGrid"))
    for _ in rows[0]:
        column = ET.SubElement(grid, q(W, "gridCol"))
        column.set(q(W, "w"), str(9000 // len(rows[0])))
    for index, row in enumerate(rows):
        tr = ET.SubElement(tbl, q(W, "tr"))
        trpr = ET.SubElement(tr, q(W, "trPr"))
        ET.SubElement(trpr, q(W, "cantSplit"))
        if index == 0:
            ET.SubElement(trpr, q(W, "tblHeader"))
        for value in row:
            tc = ET.SubElement(tr, q(W, "tc"))
            if index == 0:
                tcpr = ET.SubElement(tc, q(W, "tcPr"))
                shade = ET.SubElement(tcpr, q(W, "shd"))
                shade.set(q(W, "fill"), "E8EEF4")
            cell_p = ET.SubElement(tc, q(W, "p"))
            run = ET.SubElement(cell_p, q(W, "r"))
            if index == 0:
                ET.SubElement(ET.SubElement(run, q(W, "rPr")), q(W, "b"))
            ET.SubElement(run, q(W, "t")).text = str(value)
    return tbl


def image_paragraph(path, relationship_id, image_id):
    png = path.read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    cx = int(5.7 * 914400)
    cy = int(cx * height / width)
    p = ET.Element(q(W, "p"))
    drawing = ET.SubElement(ET.SubElement(p, q(W, "r")), q(W, "drawing"))
    inline = ET.SubElement(drawing, q(WP, "inline"))
    for side in ("distT", "distB", "distL", "distR"):
        inline.set(side, "0")
    extent = ET.SubElement(inline, q(WP, "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    docpr = ET.SubElement(inline, q(WP, "docPr"))
    docpr.set("id", str(image_id))
    docpr.set("name", path.stem)
    graphic = ET.SubElement(inline, q(A, "graphic"))
    graphic_data = ET.SubElement(graphic, q(A, "graphicData"))
    graphic_data.set("uri", PIC)
    picture = ET.SubElement(graphic_data, q(PIC, "pic"))
    nv = ET.SubElement(picture, q(PIC, "nvPicPr"))
    c_nv = ET.SubElement(nv, q(PIC, "cNvPr"))
    c_nv.set("id", "0")
    c_nv.set("name", path.name)
    ET.SubElement(nv, q(PIC, "cNvPicPr"))
    fill = ET.SubElement(picture, q(PIC, "blipFill"))
    blip = ET.SubElement(fill, q(A, "blip"))
    blip.set(q(R, "embed"), relationship_id)
    ET.SubElement(ET.SubElement(fill, q(A, "stretch")), q(A, "fillRect"))
    shape = ET.SubElement(picture, q(PIC, "spPr"))
    transform = ET.SubElement(shape, q(A, "xfrm"))
    offset = ET.SubElement(transform, q(A, "off"))
    offset.set("x", "0")
    offset.set("y", "0")
    ext = ET.SubElement(transform, q(A, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    geometry = ET.SubElement(shape, q(A, "prstGeom"))
    geometry.set("prst", "rect")
    ET.SubElement(geometry, q(A, "avLst"))
    return p


def add(nodes, heading, body):
    nodes.append(paragraph(heading, "h2", keep_next=True))
    for title, content in body:
        nodes.append(paragraph(title, "h3", keep_next=True))
        for item in content:
            nodes.append(item if isinstance(item, ET.Element) else paragraph(item))


nodes = []
add(nodes, "2.1. Problem Description", [
    ("2.1.1. Research Objective", [
        "Forecast total bike rentals for the next five consecutive hours from the preceding 24 hourly observations. The proposed model is a two-layer LSTM encoder–decoder with Bahdanau attention.",
    ]),
    ("2.1.2. Input and Output", [
        "Let t be the last observed hour. Input X has shape 24 × 19 and covers t−23 through t. Output contains five predictions for cnt(t+1) through cnt(t+5). The last observed count cnt(t) initializes the decoder; no future count is provided at inference.",
        image_paragraph(ROOT.parent / "outputs/figures/problem_formulation.png", "rId102", 102),
        paragraph("Figure 2.1. Observed 24-hour history and five forecast hours.", "caption"),
    ]),
])
add(nodes, "2.2. Dataset and Data Preprocessing", [
    ("2.2.1. Dataset Description", [
        "UCI Bike Sharing hour.csv contains 17,379 hourly records from 2011–2012. The target cnt is total rentals and equals casual + registered. Inputs contain historical demand, calendar variables, and observed weather; future weather is not used.",
    ]),
    ("2.2.2. Exploratory Data Analysis", [
        "The CSV has no blank cells or duplicate timestamps. It has 75 gaps covering 165 unrecorded hours. In training data, mean demand peaks at 17:00 (390.9) and is lowest at 04:00 (5.2). Mean cnt differs across train, validation and test: 159.9, 281.0 and 235.7. These findings motivate calendar features and chronological evaluation; they do not establish 24 hours as the best history length.",
        image_paragraph(ROOT.parent / "outputs/figures/eda_overview.png", "rId100", 100),
        paragraph("Figure 2.2. Rental distribution, training-period patterns and split means.", "caption"),
    ]),
    ("2.2.3. Feature Engineering", [
        "Sine/cosine encodings represent the cycles of hour, month and weekday. Historical cnt remains an input. Casual and registered are excluded because they sum to cnt; the row identifier and date string are also excluded. The resulting input has 19 features.",
    ]),
    ("2.2.4. Data Splitting and Normalization", [
        "After chronological sorting, rows are split 70%/15%/15% into train, validation and test. Feature and target scalers are fitted only on training rows. Later splits may use preceding rows as historical context, while their target labels remain inside their own split.",
    ]),
    ("2.2.5. Sliding Window Construction", [
        "Each sample uses 24 observed hours to forecast the following five. Windows crossing an unrecorded hour are discarded. No hourly rows are interpolated, and the supplied CSV needs no cell imputation.",
        table([["Split", "Valid forecast origins"], ["Train", "10,404"], ["Validation", "2,603"], ["Test", "2,469"], ["Total", "15,476"]]),
    ]),
])
add(nodes, "2.3. Model Architecture and Design", [
    ("2.3.1. Overall Model Architecture", [
        "The encoder turns 24 historical vectors into 24 hidden states. Attention summarizes them at each decoder step. The decoder predicts one count at a time and repeats this process five times.",
        image_paragraph(ROOT.parent / "outputs/figures/architecture_overview.png", "rId103", 103),
        paragraph("Figure 2.3. Simplified LSTM encoder, attention and decoder architecture.", "caption"),
    ]),
    ("2.3.2. LSTM Encoder", [
        "A two-layer LSTM with 64 hidden units per layer maps input (B, 24, 19) to encoder states (B, 24, 64). Its final hidden and cell states initialize the decoder.",
    ]),
    ("2.3.3. Bahdanau Attention Mechanism", [
        "At decoder step k, additive attention scores each encoder state h_i against the previous top-layer decoder state s_(k−1): e_(k,i) = v_a^T tanh(W_s s_(k−1) + W_h h_i). Softmax produces weights α_(k,i); context c_k is their weighted sum over the 24 encoder states. The implemented linear layers have no biases.",
    ]),
    ("2.3.4. LSTM Decoder", [
        "The decoder combines the preceding target or prediction with the attention context. A readout combines decoder output, context and preceding value to predict the next count. In validation and test, it uses its own earlier predictions after the first step.",
    ]),
])
add(nodes, "2.4. Training Strategy and Experimental Setup", [
    ("2.4.1. Training Configuration", [
        table([
            ["Setting", "Value"], ["Encoder / decoder layers", "2 / 2"], ["Hidden units; parameters", "64; 138,497"],
            ["Loss; optimizer", "Five-step MSE; AdamW"], ["Learning rate; weight decay", "0.001; 0.00001"],
            ["Batch size; dropout", "128; 0.2"], ["Gradient clip; initial forcing", "1.0; 0.5"],
            ["Max epochs; patience; seed", "35; 8; 42"],
        ]),
    ]),
    ("2.4.2. Teacher Forcing Strategy", [
        "During training, the decoder sometimes receives the true preceding target rather than its own prediction. The probability starts at 0.5 and declines across epochs. Validation and test use no teacher forcing.",
    ]),
    ("2.4.3. Model Optimization and Checkpoint Selection", [
        "AdamW minimizes five-step MSE. ReduceLROnPlateau lowers the learning rate when validation loss stalls. Early stopping restores the checkpoint with the lowest autoregressive validation MSE. In the verified run, that checkpoint came from epoch 5 (MSE 0.1756); training stopped at epoch 13.",
    ]),
    ("2.4.4. Training and Inference Workflow", [
        "Sort and engineer features → split and scale → construct complete windows → train and validate → restore best checkpoint → forecast test autoregressively → inverse-scale and evaluate.",
        image_paragraph(ROOT.parent / "outputs/figures/experimental_workflow.png", "rId104", 104),
        paragraph("Figure 2.4. Experimental workflow.", "caption"),
    ]),
])
add(nodes, "2.5. Evaluation Methodology", [
    ("2.5.1. Evaluation Metrics", [
        "Predictions are inverse-scaled to rentals per hour and clipped at zero. MAE = mean(|y−ŷ|); RMSE = sqrt(mean((y−ŷ)²)). Standard R² and NSE both equal 1 − Σ(y−ŷ)² / Σ(y−ȳ)² on the same observation vector, so their reported values match. Squared Pearson correlation is stored separately as R2_Pearson.",
    ]),
    ("2.5.2. Multi-horizon Evaluation", [
        "Each of five horizons is scored across 2,469 test forecast origins. Macro metrics average the five horizon scores. Overall pooled metrics use all 12,345 horizon-specific predictions. Overlapping forecast windows are not independent observations for significance tests.",
    ]),
    ("2.5.3. Baseline Methods", [
        "Persistence repeats the last observed count for all five hours. Daily seasonal naive uses the count exactly 24 hours before each forecast hour. Both use all 2,469 complete test windows. Weekly seasonal naive uses the same clock hour 168 hours earlier and is defined for 2,414 windows; 55 windows lack at least one prior-week observation.",
    ]),
])
add(nodes, "2.6. Experimental Results and Discussion", [
    ("2.6.1. Forecasting Performance by Horizon", [
        table([
            ["Horizon", "R² / NSE", "MAE", "RMSE"],
            ["t+1", "0.9408", "34.1007", "52.6394"], ["t+2", "0.9152", "40.7705", "62.8955"],
            ["t+3", "0.9052", "43.3751", "66.3678"], ["t+4", "0.8947", "45.9726", "69.9566"],
            ["t+5", "0.8781", "49.4454", "75.2998"], ["Macro", "0.9068", "42.7329", "65.4318"],
            ["Overall", "0.9069", "42.7329", "65.8717"],
        ]),
    ]),
    ("2.6.2. Performance Across Forecast Horizons", [
        "MAE rises from 34.10 at t+1 to 49.45 at t+5, while R² falls from 0.9408 to 0.8781. This is consistent with growing uncertainty and possible autoregressive error accumulation; these causes are not separated experimentally.",
    ]),
    ("2.6.3. Comparison with Baseline Methods", [
        table([["Method", "MAE", "RMSE"], ["Persistence", "166.89", "234.95"],
               ["Daily seasonal naive", "79.99", "132.60"],
               ["Weekly naive (2,414)", "64.21", "108.01"],
               ["Bahdanau (same 2,414)", "43.17", "66.33"],
               ["Original Bahdanau (2,469)", "42.73", "65.87"]]),
        "The original Bahdanau checkpoint is the main submitted model. Its full-test MAE is 42.73; on the 2,414 origins where weekly naive is defined, its MAE is 43.17 versus 64.21 for weekly naive. This comparison does not isolate the attention component.",
    ]),
    ("2.6.4. Attention Ablation (24-Hour Input)", [
        "An exploratory ablation trained zero context, fixed mean context, Bahdanau and a general-score variant with seeds 42, 43 and 44 on the same 24-hour history and test split. Values are mean ± sample standard deviation across seeds. Best-validation MSE was logged to four decimals. The general-score option uses a pre-step query, not a canonical Luong decoder.",
        table([["Context", "MAE", "RMSE", "R²", "Best val MSE"],
               ["Zero", "45.52 ± 2.68", "70.60 ± 2.91", "0.8929 ± 0.0087", "0.2165 ± 0.0217"],
               ["Fixed mean", "43.48 ± 1.67", "67.85 ± 2.53", "0.9011 ± 0.0074", "0.2100 ± 0.0275"],
               ["Bahdanau", "43.99 ± 1.53", "67.56 ± 1.82", "0.9020 ± 0.0053", "0.1885 ± 0.0113"],
               ["General score", "41.74 ± 0.52", "65.22 ± 0.31", "0.9087 ± 0.0009", "0.1841 ± 0.0090"]]),
        table([["Context", "MAE t+1", "MAE t+5"],
               ["Zero", "35.14 ± 3.63", "52.70 ± 1.67"],
               ["Fixed mean", "33.73 ± 0.77", "50.38 ± 3.18"],
               ["Bahdanau", "34.38 ± 1.93", "50.64 ± 1.08"],
               ["General score", "33.31 ± 1.32", "47.33 ± 0.36"]]),
        table([["Seed", "Zero MAE", "Mean MAE", "Bahdanau MAE", "General MAE"],
               ["42", "42.73", "42.27", "42.73", "41.98"],
               ["43", "48.06", "45.38", "43.54", "42.09"],
               ["44", "45.77", "42.78", "45.69", "41.13"]]),
        "The mean MAE gap between general score and zero context grows from 1.82 at t+1 to 5.38 at t+5, consistent with context reducing autoregressive error accumulation, but not proof of that mechanism. Fixed mean improves on zero context, while Bahdanau is not clearly better than fixed mean. The general-score result is a three-seed trend, not a significance claim.",
        "Validation and test rankings can disagree: zero context seed 42 has best-validation MSE 0.2210, worse than Bahdanau seed 43 (0.1945), but lower test MAE (42.73 versus 43.54). Selected checkpoints across all 12 runs occur at epochs 4–10, when the 35-epoch teacher-forcing schedule still used ratios about 0.37–0.46.",
        "The original Bahdanau checkpoint's average attention weights range from 0.04058 to 0.04410, close to uniform 1/24 = 0.04167. In the seed-42 ablation, Bahdanau averages range from 0.04058 to 0.04411; general-score averages range from 0.02572 to 0.07361. Fixed mean performs similarly to Bahdanau in average test MAE, so this experiment does not show a clear gain from nonuniform Bahdanau weights.",
        "All four variants have already been inspected on the test period. A future final configuration must be chosen by validation before evaluation on a fresh test period. The original Bahdanau checkpoint remains the main submitted model.",
    ]),
    ("2.6.5. Forecast Visualization", [
        "Actual and predicted counts at horizons t+1 and t+5 are shown for the first 168 valid test forecast origins. The x-axis indexes origins, which may skip clock hours after gap-crossing windows are removed.",
        image_paragraph(ROOT.parent / "outputs/figures/forecast_comparison.png", "rId101", 101),
        paragraph("Figure 2.5. Test forecasts at the first and fifth horizons.", "caption"),
    ]),
    ("2.6.6. Limitations and Discussion", [
        "The experiment uses one dataset and one chronological test period. Demand differs across periods, so accuracy may not transfer to another year or city. The context ablation uses only three seeds and a 24-hour input. The test set has already been inspected across variants, so selecting a final model now would require a fresh test period. Overlapping windows also limit independence assumptions for confidence intervals.",
        "An earlier exploratory daily-data run under outputs/day/ has pooled R² 0.1668 and MAE 1,497.48 rentals/day. It has no checkpoint metadata and is not treated as a comparable result for the hourly task.",
    ]),
])
add(nodes, "2.7. Conclusion", [
    ("Summary", [
        "The original Bahdanau model forecasts five hours from a 24-hour history. Across 2,469 valid test origins, pooled MAE is 42.73, RMSE is 65.87 and R² is 0.9069, below persistence and daily seasonal naive. On the 2,414 origins with a valid weekly lag, it also beats weekly naive. In the exploratory three-seed ablation, general score has the lowest mean MAE (41.74), followed by fixed mean (43.48), Bahdanau (43.99) and zero context (45.52). This does not select a new final model.",
    ]),
])


with ZipFile(SOURCE) as zin:
    source_xml = zin.read("word/document.xml")
    document = ET.fromstring(source_xml)
    body = document.find(q(W, "body"))
    children = list(body)
    start = next(i for i, node in enumerate(children) if node.tag == q(W, "p") and text_of(node).startswith("2. Task 2"))
    end = next(i for i, node in enumerate(children) if node.tag == q(W, "p") and text_of(node).startswith("3. Task 3"))
    for node in children[start + 1:end]:
        body.remove(node)
    for offset, node in enumerate(nodes, 1):
        body.insert(start + offset, node)

    # The supplied template already has a References section after all tasks.
    refs_heading = next(node for node in body if node.tag == q(W, "p") and text_of(node) == "References")
    refs_at = list(body).index(refs_heading) + 1
    placeholder = list(body)[refs_at]
    if placeholder.tag == q(W, "p") and text_of(placeholder).startswith("List all references cited"):
        body.remove(placeholder)
    references = [
        "Fanaee-T, H., and Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2, 113–127. UCI Bike Sharing dataset 275.",
        "Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. arXiv:1409.0473.",
        "Nash, J. E., and Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I. Journal of Hydrology, 10(3), 282–290.",
    ]
    for offset, reference in enumerate(references):
        body.insert(refs_at + offset, paragraph(reference))

    xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    # ElementTree drops declarations used only in mc:Ignorable; Word needs them.
    source_root_start = source_xml.index(b"<w:document")
    source_root_end = source_xml.index(b">", source_root_start)
    namespaces = re.findall(rb'xmlns:([\w-]+)="([^"]+)"', source_xml[source_root_start:source_root_end])
    root_end = xml.index(b">", xml.index(b"<w:document"))
    root_tag = xml[:root_end]
    missing = b"".join(b' xmlns:' + prefix + b'="' + uri + b'"' for prefix, uri in namespaces
                       if b'xmlns:' + prefix + b'=' not in root_tag)
    xml = xml[:root_end] + missing + xml[root_end:]

    rels = ET.fromstring(zin.read("word/_rels/document.xml.rels"))
    images = (("rId100", ROOT.parent / "outputs/figures/eda_overview.png"),
              ("rId101", ROOT.parent / "outputs/figures/forecast_comparison.png"),
              ("rId102", ROOT.parent / "outputs/figures/problem_formulation.png"),
              ("rId103", ROOT.parent / "outputs/figures/architecture_overview.png"),
              ("rId104", ROOT.parent / "outputs/figures/experimental_workflow.png"))
    for rel_id, image_path in images:
        relation = ET.SubElement(rels, q(PKG_R, "Relationship"))
        relation.set("Id", rel_id)
        relation.set("Type", R + "/image")
        relation.set("Target", "media/" + image_path.name)
    types = ET.fromstring(zin.read("[Content_Types].xml"))
    if not any(item.get("Extension") == "png" for item in types):
        item = ET.SubElement(types, q(CT, "Default"))
        item.set("Extension", "png")
        item.set("ContentType", "image/png")
    with ZipFile(DEST, "w", ZIP_DEFLATED) as zout:
        for member in zin.infolist():
            content = {"word/document.xml": xml,
                       "word/_rels/document.xml.rels": ET.tostring(rels, encoding="utf-8", xml_declaration=True),
                       "[Content_Types].xml": ET.tostring(types, encoding="utf-8", xml_declaration=True)}.get(
                           member.filename, zin.read(member.filename))
            zout.writestr(member, content)
        for _, image_path in images:
            zout.write(image_path, "word/media/" + image_path.name)

print(DEST)
