fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let swift_target = match arch.as_str() {
            "aarch64" => "arm64-apple-macos13.0",
            "x86_64" => "x86_64-apple-macos13.0",
            other => panic!("unsupported macOS OCR architecture: {other}"),
        };
        let cache = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("swift-cache");
        let out = std::path::Path::new("resources/native-ocr");
        std::fs::create_dir_all(out).expect("create bundled OCR directory");
        for (source, binary) in [
            ("vision_ocr-step5.swift", "local-search-ocr-step5"),
            ("pdf_vision-step6.swift", "local-search-pdf-ocr-step6"),
        ] {
            let source = std::path::Path::new("native-ocr").join(source);
            println!("cargo:rerun-if-changed={}", source.display());
            let status = std::process::Command::new("xcrun")
                .args(["swiftc", "-O", "-target", swift_target])
                .arg("-module-cache-path").arg(&cache)
                .arg(&source).arg("-o").arg(out.join(binary))
                .status().expect("xcrun swiftc required to build native OCR");
            assert!(status.success(), "failed to compile {}", source.display());
        }
    }
    tauri_build::build()
}
