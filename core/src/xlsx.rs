//! XLSX text extraction for local search. Bounded ZIP member reads; no external processes.
//! Preserves sheet identity, cell coordinates, shared/inline strings and numeric results.
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use zip::ZipArchive;

const MAX_XML_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SHEETS: usize = 96;
const MAX_SHARED_STRINGS: usize = 250_000;

fn local(tag: &[u8]) -> &[u8] {
    tag.rsplit(|x| *x == b':').next().unwrap_or(tag)
}

fn attr(tag: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    tag.attributes().with_checks(false).flatten().find_map(|a| {
        if local(a.key.as_ref()) == name {
            std::str::from_utf8(a.value.as_ref())
                .ok()
                .and_then(|raw| quick_xml::escape::unescape(raw).ok())
                .map(|v| v.into_owned())
        } else {
            None
        }
    })
}

fn xml_text(e: &quick_xml::events::BytesText<'_>) -> String {
    let decoded = e.decode().unwrap_or_default();
    quick_xml::escape::unescape(&decoded)
        .map(|v| v.into_owned())
        .unwrap_or_else(|_| decoded.to_string())
}

fn member<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Option<String> {
    let mut f = archive.by_name(name).ok()?;
    if f.size() > MAX_XML_BYTES {
        return None;
    }
    let mut buf = Vec::new();
    f.by_ref()
        .take(MAX_XML_BYTES + 1)
        .read_to_end(&mut buf)
        .ok()?;
    if buf.len() as u64 > MAX_XML_BYTES {
        return None;
    }
    String::from_utf8(buf).ok()
}

// quick-xml >= 0.41 emits &amp; and numeric references as GeneralRef,
// separately from Text. Losing those events silently corrupts searchable text.
fn resolve_xml_reference(e: &quick_xml::events::BytesRef<'_>) -> String {
    if let Ok(Some(ch)) = e.resolve_char_ref() {
        return ch.to_string();
    }
    if let Ok(name) = e.decode() {
        if let Some(value) = quick_xml::escape::resolve_predefined_entity(&name) {
            return value.to_string();
        }
        // Unknown general references are preserved literally, never expanded.
        return format!("&{name};");
    }
    String::new()
}

fn shared_strings(xml: &str) -> Vec<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut strings = Vec::new();
    let mut value = String::new();
    let mut in_si = false;
    let mut in_t = false;
    let mut phonetic = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                b"si" => {
                    in_si = true;
                    value.clear();
                }
                b"rPh" => phonetic = true,
                b"t" if in_si && !phonetic => in_t = true,
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"si" => {
                    strings.push(std::mem::take(&mut value));
                    in_si = false;
                    if strings.len() >= MAX_SHARED_STRINGS {
                        break;
                    }
                }
                b"rPh" => phonetic = false,
                b"t" => in_t = false,
                _ => {}
            },
            Ok(Event::Text(e)) if in_si && in_t => value.push_str(&xml_text(&e)),
            Ok(Event::GeneralRef(e)) if in_si && in_t => value.push_str(&resolve_xml_reference(&e)),
            Ok(Event::CData(e)) if in_si && in_t => {
                value.push_str(&String::from_utf8_lossy(e.as_ref()))
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    strings
}

fn workbook_sheets(xml: &str) -> Vec<(String, String)> {
    let mut reader = Reader::from_str(xml);
    let mut sheets = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) if local(e.name().as_ref()) == b"sheet" => {
                if let (Some(name), Some(id)) = (attr(&e, b"name"), attr(&e, b"id")) {
                    sheets.push((name, id));
                    if sheets.len() >= MAX_SHEETS {
                        break;
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    sheets
}

fn workbook_relationships(xml: &str) -> HashMap<String, String> {
    let mut reader = Reader::from_str(xml);
    let mut rels = HashMap::new();
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e))
                if local(e.name().as_ref()) == b"Relationship" =>
            {
                if let (Some(id), Some(target), Some(kind)) =
                    (attr(&e, b"Id"), attr(&e, b"Target"), attr(&e, b"Type"))
                {
                    if kind.ends_with("/worksheet") {
                        // Reject path traversal and non-worksheet members.
                        let path = if let Some(p) = target.strip_prefix("/xl/") {
                            format!("xl/{p}")
                        } else if target.starts_with("xl/") {
                            target
                        } else {
                            format!("xl/{target}")
                        };
                        if path.starts_with("xl/worksheets/")
                            && !path.contains("..")
                            && path.ends_with(".xml")
                        {
                            rels.insert(id, path);
                        }
                    }
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    rels
}

fn emit_cell(
    out: &mut String,
    cap: usize,
    coord: &str,
    kind: &str,
    raw: &str,
    formula: &str,
    inline: &str,
    shared: &[String],
) {
    let val = match kind {
        "s" => raw
            .trim()
            .parse::<usize>()
            .ok()
            .and_then(|i| shared.get(i))
            .map(String::as_str)
            .unwrap_or(""),
        "inlineStr" => inline,
        "b" => {
            if raw.trim() == "1" {
                "TRUE"
            } else {
                "FALSE"
            }
        }
        _ => raw,
    };
    if val.is_empty() && formula.is_empty() {
        return;
    }
    let item = if formula.is_empty() {
        val.to_string()
    } else if val.is_empty() {
        format!("={formula}")
    } else {
        format!("={formula} => {val}")
    };
    // Keep lines and coordinates so numeric cells remain searchable.
    let fragment = format!("{coord}: {item}\n");
    if out.len().saturating_add(fragment.len()) <= cap {
        out.push_str(&fragment);
    }
}

fn sheet_text(xml: &str, shared: &[String], cap: usize) -> String {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut out = String::new();
    let (mut coord, mut kind, mut raw, mut formula, mut inline) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    let (mut in_cell, mut in_value, mut in_formula, mut in_inline, mut in_text) =
        (false, false, false, false, false);
    let mut row_count = 0_usize;
    loop {
        if out.len() >= cap.saturating_sub(64) || row_count >= 100_000 {
            break;
        }
        match reader.read_event() {
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                b"c" => {
                    in_cell = true;
                    coord = attr(&e, b"r").unwrap_or_default();
                    kind = attr(&e, b"t").unwrap_or_default();
                    raw.clear();
                    formula.clear();
                    inline.clear();
                }
                b"v" if in_cell => in_value = true,
                b"f" if in_cell => in_formula = true,
                b"is" if in_cell => in_inline = true,
                b"t" if in_cell && in_inline => in_text = true,
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                b"c" if in_cell => {
                    emit_cell(
                        &mut out, cap, &coord, &kind, &raw, &formula, &inline, shared,
                    );
                    in_cell = false;
                }
                b"v" => in_value = false,
                b"f" => in_formula = false,
                b"is" => in_inline = false,
                b"t" => in_text = false,
                b"row" => row_count += 1,
                _ => {}
            },
            Ok(Event::Text(e)) if in_cell => {
                let text = xml_text(&e);
                if in_value {
                    raw.push_str(&text);
                } else if in_formula {
                    formula.push_str(&text);
                } else if in_text {
                    inline.push_str(&text);
                }
            }
            Ok(Event::GeneralRef(e)) if in_cell => {
                let reference = resolve_xml_reference(&e);
                if in_value {
                    raw.push_str(&reference);
                } else if in_formula {
                    formula.push_str(&reference);
                } else if in_text {
                    inline.push_str(&reference);
                }
            }
            Ok(Event::CData(e)) if in_cell && in_text => {
                inline.push_str(&String::from_utf8_lossy(e.as_ref()))
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

pub fn extract_xlsx_text(path: &Path, char_cap: usize) -> Option<String> {
    let f = File::open(path).ok()?;
    let mut zip = ZipArchive::new(f).ok()?;
    let strings = member(&mut zip, "xl/sharedStrings.xml")
        .map(|x| shared_strings(&x))
        .unwrap_or_default();
    let names = member(&mut zip, "xl/workbook.xml")
        .map(|x| workbook_sheets(&x))
        .unwrap_or_default();
    let rels = member(&mut zip, "xl/_rels/workbook.xml.rels")
        .map(|x| workbook_relationships(&x))
        .unwrap_or_default();

    let mut sheets: Vec<(String, String)> = names
        .into_iter()
        .filter_map(|(name, id)| rels.get(&id).map(|path| (name, path.to_string())))
        .collect();
    if sheets.is_empty() {
        let mut paths: Vec<String> = zip
            .file_names()
            .filter(|n| {
                n.starts_with("xl/worksheets/sheet") && n.ends_with(".xml") && !n.contains("..")
            })
            .take(MAX_SHEETS)
            .map(String::from)
            .collect();
        paths.sort();
        sheets = paths
            .into_iter()
            .map(|p| (p.rsplit('/').next().unwrap_or("Sheet").to_string(), p))
            .collect();
    }
    let cap = char_cap.min(2_000_000).max(1);
    let mut text = String::new();
    for (name, path) in sheets.into_iter().take(MAX_SHEETS) {
        if text.len() >= cap.saturating_sub(64) {
            break;
        }
        if let Some(xml) = member(&mut zip, &path) {
            let prefix = format!("Лист: {name}\n");
            if text.len() + prefix.len() > cap {
                break;
            }
            text.push_str(&prefix);
            let remaining = cap - text.len();
            text.push_str(&sheet_text(&xml, &strings, remaining));
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn shared_strings_preserve_rich_text_and_entities() {
        let v = shared_strings("<sst><si><t>Редуктор &amp; масло</t></si><si><r><t>Масло</t></r><r><t> замена</t></r><rPh><t>ignore</t></rPh></si></sst>");
        assert_eq!(v, vec!["Редуктор & масло", "Масло замена"]);
    }

    #[test]
    fn shared_strings_decode_numeric_and_predefined_entities() {
        let entries =
            shared_strings("<sst><si><t>ТО &#8470;1 &amp; замена &lt;масла&gt;</t></si></sst>");
        assert_eq!(entries, vec!["ТО №1 & замена <масла>"]);
    }

    #[test]
    fn inline_strings_preserve_xml_entities() {
        let xml = "<worksheet><sheetData><row><c r=\"A1\" t=\"inlineStr\"><is><t>Масло &amp; фильтр &#8470;2</t></is></c></row></sheetData></worksheet>";
        let text = sheet_text(xml, &[], 500);
        assert!(text.contains("A1: Масло & фильтр №2"), "{text}");
    }

    #[test]
    fn sheet_cells_keep_strings_numbers_formulas_coordinates() {
        let sheet = r#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>68000</v></c><c r="C1" t="inlineStr"><is><t>Редуктор</t></is></c><c r="D1"><f>B1*2</f><v>136000</v></c><c r="E1" t="b"><v>1</v></c></row></sheetData></worksheet>"#;
        let text = sheet_text(sheet, &["Замена масла".into()], 2000);
        for want in [
            "A1: Замена масла",
            "B1: 68000",
            "C1: Редуктор",
            "D1: =B1*2 => 136000",
            "E1: TRUE",
        ] {
            assert!(text.contains(want), "missing {want} in {text}");
        }
    }

    #[test]
    fn end_to_end_xlsx_zip() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::<u8>::new()));
        let options = zip::write::SimpleFileOptions::default();
        let entries = [
            (
                "xl/workbook.xml",
                r#"<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="ТО автомобиля" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
            ),
            (
                "xl/_rels/workbook.xml.rels",
                r#"<Relationships><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
            ),
            (
                "xl/sharedStrings.xml",
                "<sst><si><t>Масло редуктора</t></si></sst>",
            ),
            (
                "xl/worksheets/sheet1.xml",
                r#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>68000</v></c></row></sheetData></worksheet>"#,
            ),
        ];
        for (name, content) in entries {
            writer.start_file(name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        let bytes = writer.finish().unwrap().into_inner();
        let tmp = std::env::temp_dir().join(format!(
            "local_search_{}_{}.xlsx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&tmp, bytes).unwrap();
        let txt = extract_xlsx_text(&tmp, 10000).unwrap();
        std::fs::remove_file(&tmp).unwrap();
        assert!(txt.contains("Лист: ТО автомобиля"));
        assert!(txt.contains("A1: Масло редуктора"));
        assert!(txt.contains("B1: 68000"));
    }
}
