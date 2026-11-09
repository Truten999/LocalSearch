// LocalSearch Vision OCR bridge — macOS only. No network, no third-party models.
// Apache/MIT-compatible independent code written for this project.
import Foundation
import AppKit
import Vision
import ImageIO
import UniformTypeIdentifiers
import Darwin

private enum OCRError: LocalizedError {
    case usage
    case fileMissing(String)
    case fileTooLarge(Int64)
    case unsupportedImage(String)
    case unsupportedRussian([String])
    case imageDecode(String)
    case encodeFailed(String)
    case noText(String)
    case incorrectOCR(String, String)

    var errorDescription: String? {
        switch self {
        case .usage: return "Usage: local-search-ocr --languages | --self-test | --ocr <image>"
        case .fileMissing(let path): return "File missing: \(path)"
        case .fileTooLarge(let size): return "Input image exceeds 80 MiB (\(size) bytes)"
        case .unsupportedImage(let ext): return "Unsupported image extension: \(ext)"
        case .unsupportedRussian(let languages): return "Vision Russian unavailable. Languages: \(languages.joined(separator: ", "))"
        case .imageDecode(let path): return "ImageIO could not decode: \(path)"
        case .encodeFailed(let ext): return "ImageIO could not encode test image: \(ext)"
        case .noText(let path): return "Vision returned no text for: \(path)"
        case .incorrectOCR(let path, let text): return "Unreliable Russian OCR for \(path): \(text)"
        }
    }
}

private let allowedExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "webp", "bmp", "gif"]
private let maxInputBytes: Int64 = 80 * 1024 * 1024
private let maxDimension = 2400

private func recognitionRequest() throws -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    // Probe the installed OS revision; don't assume its language list.
    let supported = try request.supportedRecognitionLanguages()
    guard let ru = supported.first(where: { $0.lowercased().hasPrefix("ru") }) else {
        throw OCRError.unsupportedRussian(supported)
    }
    let en = supported.first(where: { $0.lowercased().hasPrefix("en") })
    request.recognitionLanguages = [ru] + (en == nil ? [] : [en!])
    request.usesLanguageCorrection = true
    return request
}

private func decodeBounded(_ url: URL) throws -> CGImage {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw OCRError.fileMissing(url.path)
    }
    let ext = url.pathExtension.lowercased()
    guard allowedExtensions.contains(ext) else { throw OCRError.unsupportedImage(ext) }
    let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
    let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
    guard size <= maxInputBytes else { throw OCRError.fileTooLarge(size) }
    guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else {
        throw OCRError.imageDecode(url.path)
    }
    let opts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxDimension,
        kCGImageSourceShouldCacheImmediately: true
    ]
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, opts as CFDictionary) else {
        throw OCRError.imageDecode(url.path)
    }
    return image
}

private func extractText(_ url: URL) throws -> String {
    let image = try decodeBounded(url)
    let request = try recognitionRequest()
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
}

private func makeTestImage() throws -> CGImage {
    let width = 1800
    let height = 480
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
    ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw OCRError.encodeFailed("bitmap")
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: width, height: height).fill()
    let attrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 78, weight: .semibold),
        .foregroundColor: NSColor.black
    ]
    ("Замена масла в редукторе" as NSString).draw(at: NSPoint(x: 75, y: 260), withAttributes: attrs)
    ("Техническое обслуживание" as NSString).draw(at: NSPoint(x: 75, y: 115), withAttributes: attrs)
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let image = bitmap.cgImage else { throw OCRError.encodeFailed("CGImage") }
    return image
}

private func writeImage(_ image: CGImage, to url: URL, type: UTType) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
        throw OCRError.encodeFailed(type.identifier)
    }
    CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
    guard CGImageDestinationFinalize(dest) else { throw OCRError.encodeFailed(type.identifier) }
}

private func runSelfTest() throws {
    // Both source images are locally generated; no access to user's archive.
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("localsearch-ocr-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let sample = try makeTestImage()
    for (ext, type) in [("png", UTType.png), ("heic", UTType.heic)] {
        let path = dir.appendingPathComponent("russian.\(ext)")
        try writeImage(sample, to: path, type: type)
        let text = try extractText(path).lowercased()
        // Deliberately check meaning-bearing Cyrillic words, not just nonempty output.
        guard text.contains("масла") && text.contains("редуктор") else {
            throw OCRError.incorrectOCR(ext, text)
        }
        print("\(ext.uppercased())_RUSSIAN_OCR_OK")
    }
}

private func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    if args.count == 1 && args[0] == "--languages" {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        print(try request.supportedRecognitionLanguages().joined(separator: ", "))
    } else if args.count == 1 && args[0] == "--self-test" {
        try runSelfTest()
        print("VISION_SELFTEST_OK")
    } else if args.count == 2 && args[0] == "--make-fixtures" {
        let directory = URL(fileURLWithPath: args[1], isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let image = try makeTestImage()
        try writeImage(image, to: directory.appendingPathComponent("sample.png"), type: .png)
        try writeImage(image, to: directory.appendingPathComponent("sample.heic"), type: .heic)
        print("VISION_FIXTURES_OK")
    } else if args.count == 2 && args[0] == "--ocr" {
        let file = URL(fileURLWithPath: args[1])
        let text = try extractText(file)
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw OCRError.noText(file.path) }
        print(text)
    } else {
        throw OCRError.usage
    }
}

do {
    try run()
} catch {
    fputs("STEP4_ERROR: \(error.localizedDescription)\n", stderr)
    exit(1)
}
