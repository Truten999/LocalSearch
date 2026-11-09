// macOS Apple Vision OCR for scanned PDFs. Fully local, no Python dependencies.
// Intentionally limited to small scanned PDFs until paginated ingestion is added.
import Foundation
import AppKit
import Vision
import PDFKit

private enum PDFOCRError: LocalizedError {
    case usage
    case missing(String)
    case oversized(Int64)
    case unreadable(String)
    case pages(Int)
    case russianUnavailable
    case render(Int)
    case empty
    case fixture

    var errorDescription: String? {
        switch self {
        case .usage: return "Usage: pdf-vision --ocr-pdf <path> | --make-pdf-fixture <directory> | --self-test"
        case .missing(let path): return "Missing PDF: \(path)"
        case .oversized(let bytes): return "PDF too large for this bounded test: \(bytes) bytes"
        case .unreadable(let path): return "Cannot open PDF: \(path)"
        case .pages(let count): return "PDF has \(count) pages; bounded OCR accepts 1...32. File left pending."
        case .russianUnavailable: return "Russian is unavailable in Apple Vision"
        case .render(let page): return "Cannot render PDF page \(page)"
        case .empty: return "No OCR text in scanned PDF"
        case .fixture: return "Unable to create synthetic scanned PDF"
        }
    }
}

private let maxPDFBytes: Int64 = 96 * 1024 * 1024
private let maxPages = 32
private let rasterLongEdge: CGFloat = 2200

private func newTextRequest() throws -> VNRecognizeTextRequest {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    let available = try req.supportedRecognitionLanguages()
    guard let russian = available.first(where: { $0.lowercased().hasPrefix("ru") }) else {
        throw PDFOCRError.russianUnavailable
    }
    let english = available.first(where: { $0.lowercased().hasPrefix("en") })
    req.recognitionLanguages = [russian] + (english.map { [$0] } ?? [])
    req.usesLanguageCorrection = true
    return req
}

private func recognizePage(_ page: PDFPage, index: Int) throws -> String {
    let bounds = page.bounds(for: .mediaBox)
    guard bounds.width > 0 && bounds.height > 0 else { throw PDFOCRError.render(index) }
    let ratio = rasterLongEdge / max(bounds.width, bounds.height)
    let size = NSSize(width: max(1, bounds.width * ratio), height: max(1, bounds.height * ratio))
    let thumbnail = page.thumbnail(of: size, for: .mediaBox)
    var proposed = NSRect(origin: .zero, size: thumbnail.size)
    guard let image = thumbnail.cgImage(forProposedRect: &proposed, context: nil, hints: nil) else {
        throw PDFOCRError.render(index)
    }
    let request = try newTextRequest()
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

private func recognizePDF(_ url: URL) throws -> String {
    guard FileManager.default.fileExists(atPath: url.path) else { throw PDFOCRError.missing(url.path) }
    let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
    let count = (attrs[.size] as? NSNumber)?.int64Value ?? 0
    guard count <= maxPDFBytes else { throw PDFOCRError.oversized(count) }
    guard let pdf = PDFDocument(url: url) else { throw PDFOCRError.unreadable(url.path) }
    guard pdf.pageCount > 0 && pdf.pageCount <= maxPages else { throw PDFOCRError.pages(pdf.pageCount) }
    var collected: [String] = []
    // Only scanned-image documents (no existing useful text) are passed here by Rust.
    // Keep page boundaries; no silent truncation.
    for i in 0..<pdf.pageCount {
        guard let page = pdf.page(at: i) else { throw PDFOCRError.render(i + 1) }
        let text = try recognizePage(page, index: i + 1).trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty { collected.append("Страница \(i + 1)\n\(text)") }
    }
    guard !collected.isEmpty else { throw PDFOCRError.empty }
    return collected.joined(separator: "\n\n")
}

private func makeFixture(in directory: URL) throws -> URL {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 1800, pixelsHigh: 700,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
    )
    guard let bitmap, let ctx = NSGraphicsContext(bitmapImageRep: bitmap) else { throw PDFOCRError.fixture }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = ctx
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: 1800, height: 700).fill()
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 87, weight: .semibold),
        .foregroundColor: NSColor.black
    ]
    ("Замена масла в редукторе" as NSString).draw(at: NSPoint(x: 70, y: 410), withAttributes: attrs)
    ("Техническое обслуживание" as NSString).draw(at: NSPoint(x: 70, y: 230), withAttributes: attrs)
    ctx.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let cg = bitmap.cgImage else { throw PDFOCRError.fixture }
    let pdf = PDFDocument()
    let image = NSImage(cgImage: cg, size: NSSize(width: 900, height: 350))
    guard let page = PDFPage(image: image) else { throw PDFOCRError.fixture }
    pdf.insert(page, at: 0)
    let path = directory.appendingPathComponent("scan-russian.pdf")
    guard pdf.write(to: path) else { throw PDFOCRError.fixture }
    // A scanned PDF must not carry an accidental text layer.
    guard let reopened = PDFDocument(url: path),
          (reopened.page(at: 0)?.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { throw PDFOCRError.fixture }
    return path
}

private func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    if args.count == 2 && args[0] == "--ocr-pdf" {
        print(try recognizePDF(URL(fileURLWithPath: args[1])))
    } else if args.count == 2 && args[0] == "--make-pdf-fixture" {
        let path = try makeFixture(in: URL(fileURLWithPath: args[1], isDirectory: true))
        print("SCANNED_PDF_FIXTURE_OK \(path.lastPathComponent)")
    } else if args.count == 1 && args[0] == "--self-test" {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ls-pdf-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = try makeFixture(in: dir)
        let result = try recognizePDF(path).lowercased()
        guard result.contains("масла") && result.contains("редуктор") else { throw PDFOCRError.fixture }
        print("PDF_RUSSIAN_OCR_OK")
    } else {
        throw PDFOCRError.usage
    }
}

do {
    try main()
} catch {
    fputs("PDF_VISION_ERROR: \(error.localizedDescription)\n", stderr)
    exit(1)
}
