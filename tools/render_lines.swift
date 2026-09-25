// Arap harfli metin satırlarını CoreText ile (doğru şekillendirme + RTL)
// PNG sayfalara çizer. PIL'de libraqm olmadığı için gerekli.
//
// Kullanım:
//   xcrun swiftc -O -o tools/render_lines tools/render_lines.swift
//   tools/render_lines <metin.txt> <çıktı_dizini> [font=DecoTypeNaskh] [punto=44] [satır/sayfa=6]
// Her sayfa: <çıktı>/pageNN.png + pageNN.gt.txt (aynı satırlar).
import AppKit

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("kullanım: render_lines <metin.txt> <çıktı_dizini> [font] [punto] [satır/sayfa]\n".data(using: .utf8)!)
    exit(2)
}
let textPath = args[1]
let outDir = args[2]
let fontName = args.count > 3 ? args[3] : "DecoTypeNaskh"
let fontSize = args.count > 4 ? CGFloat(Double(args[4]) ?? 44) : 44
let linesPerPage = args.count > 5 ? Int(args[5]) ?? 6 : 6

guard let font = NSFont(name: fontName, size: fontSize) else {
    FileHandle.standardError.write("font bulunamadı: \(fontName)\n".data(using: .utf8)!)
    exit(3)
}
let raw = try! String(contentsOfFile: textPath, encoding: .utf8)
let lines = raw.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
try! FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

let pageW: CGFloat = 1800
let margin: CGFloat = 90
let lineGap: CGFloat = fontSize * 2.2

let para = NSMutableParagraphStyle()
para.alignment = .right
para.baseWritingDirection = .rightToLeft
let attrs: [NSAttributedString.Key: Any] = [
    .font: font, .foregroundColor: NSColor.black, .paragraphStyle: para
]

var pageIdx = 0
var i = 0
while i < lines.count {
    let chunk = Array(lines[i..<min(i + linesPerPage, lines.count)])
    let pageH = margin * 2 + lineGap * CGFloat(chunk.count)
    let img = NSImage(size: NSSize(width: pageW, height: pageH))
    img.lockFocus()
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: pageW, height: pageH).fill()
    for (k, line) in chunk.enumerated() {
        // AppKit koordinatı alttan başlar; üstten k. satır
        let y = pageH - margin - lineGap * CGFloat(k + 1) + (lineGap - fontSize) / 2
        let rect = NSRect(x: margin, y: y, width: pageW - 2 * margin, height: lineGap)
        NSAttributedString(string: line, attributes: attrs).draw(in: rect)
    }
    img.unlockFocus()
    guard let tiff = img.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else {
        FileHandle.standardError.write("png üretilemedi\n".data(using: .utf8)!)
        exit(4)
    }
    let stem = String(format: "page%02d", pageIdx)
    try! png.write(to: URL(fileURLWithPath: "\(outDir)/\(stem).png"))
    try! (chunk.joined(separator: "\n") + "\n").write(toFile: "\(outDir)/\(stem).gt.txt", atomically: true, encoding: .utf8)
    pageIdx += 1
    i += linesPerPage
}
print("\(pageIdx) sayfa yazıldı: \(outDir) (font=\(fontName), punto=\(Int(fontSize)))")
