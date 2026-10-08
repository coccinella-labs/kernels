import XCTest

/// Guards README claims against the implementation.
///
/// The kernel count, the list of undispatched kernels, and the Metal/CUDA code
/// samples are all checkable from the repository itself. This test fails when
/// the README and the sources disagree, so documentation drift breaks the build
/// rather than accumulating.
final class ReadmeDriftTests: XCTestCase {

    private func repoRoot() -> URL {
        // Tests/MetalKernelsTests/<file>.swift -> repository root
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func readme() -> String {
        let url = repoRoot().appendingPathComponent("README.md")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            XCTFail("README.md is missing or unreadable")
            return ""
        }
        return text
    }

    private func source(_ relative: String) -> String {
        let url = repoRoot().appendingPathComponent(relative)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            XCTFail("\(relative) is missing")
            return ""
        }
        return text
    }

    /// Kernel names declared in the Metal source.
    private func declaredKernels() -> [String] {
        let metal = source("Sources/MetalKernels/kernels.metal")
        var names: [String] = []
        for line in metal.split(separator: "\n").map(String.init) {
            guard line.hasPrefix("kernel void ") else { continue }
            let rest = line.dropFirst("kernel void ".count)
            names.append(String(rest.prefix { $0.isLetter || $0.isNumber || $0 == "_" }))
        }
        return names.sorted()
    }

    /// Kernels the demo actually dispatches by name.
    private func dispatchedKernels() -> Set<String> {
        let swift = source("Sources/MetalKernels/main.swift")
        let all = declaredKernels()
        return Set(all.filter { swift.contains("\"\($0)\"") })
    }

    /// The kernel count the README states must equal the number in the source.
    func testKernelCountClaimMatchesSource() {
        let kernels = declaredKernels()
        XCTAssertFalse(kernels.isEmpty, "no kernels parsed from kernels.metal")

        // Accept any phrasing where a count sits next to the word "kernel", so
        // rewording the sentence is fine but changing the number fails CI.
        let readme = self.readme()
        let pattern = try! NSRegularExpression(pattern: "([0-9]+) +(?:[Mm]etal +)?(?:compute +)?kernels?")
        let ns = readme as NSString
        let counts = pattern.matches(in: readme, range: NSRange(location: 0, length: ns.length))
            .compactMap { Int(ns.substring(with: $0.range(at: 1))) }

        XCTAssertFalse(
            counts.isEmpty,
            "README states no kernel count; the extraction pattern is broken"
        )
        let wrong = counts.filter { $0 != kernels.count }
        XCTAssertTrue(
            wrong.isEmpty,
            "README states kernel counts \(wrong) but kernels.metal declares \(kernels.count). Update every mention."
        )
    }

    /// Every kernel the README names as undispatched really is undispatched, and
    /// the list is complete.
    func testUndispatchedKernelListIsAccurate() {
        let undispatched = Set(declaredKernels()).subtracting(dispatchedKernels())
        XCTAssertFalse(
            undispatched.isEmpty,
            "expected some kernels to be shader-only"
        )

        let readme = self.readme()
        let section = readme
            .components(separatedBy: "## ")
            .first { $0.hasPrefix("Kernels in this repository") } ?? readme

        // Every kernel the README claims is undispatched must actually be one.
        for kernel in undispatched {
            XCTAssertTrue(
                section.contains(kernel),
                "kernel `\(kernel)` is never dispatched by the demo but the README does not say so"
            )
        }

        // The reverse: nothing the README lists as shader-only may be dispatched.
        let shaderOnly = section
            .components(separatedBy: "| `")
            .dropFirst()
            .compactMap { $0.components(separatedBy: "`").first }
            .filter { name in !name.contains(" ") }

        // Every name in the table must exist in kernels.metal. Without this a
        // typo or a hallucinated kernel would slip through, because such a name
        // is neither dispatched nor undispatched.
        let declared = Set(declaredKernels())
        let invented = shaderOnly.filter { !declared.contains($0) }
        XCTAssertTrue(
            invented.isEmpty,
            "README lists kernels that do not exist in kernels.metal: \(invented.sorted())"
        )

        let listedUndispatched = shaderOnly.filter { undispatched.contains($0) }
        XCTAssertEqual(
            listedUndispatched.count,
            undispatched.count,
            "README lists \(listedUndispatched.count) shader-only kernels, source has \(undispatched.count): \(undispatched.sorted())"
        )

        // And no dispatched kernel may be described as shader-only.
        let wronglyListed = shaderOnly.filter { dispatchedKernels().contains($0) }
        XCTAssertTrue(
            wronglyListed.isEmpty,
            "README calls dispatched kernels shader-only: \(wronglyListed.sorted())"
        )
    }

    /// The CUDA and Metal samples in the README must match the real kernel.
    func testCodeSamplesMatchTheRealKernel() {
        let readme = self.readme()
        let metal = source("Sources/MetalKernels/kernels.metal")

        // The README presents vector_add as the worked example; both snippets
        // must agree with the actual source.
        guard let start = metal.range(of: "kernel void vector_add") else {
            return XCTFail("vector_add not found in kernels.metal")
        }
        let tail = metal[start.upperBound...]
        let end = tail.range(of: "\n}")?.lowerBound ?? tail.endIndex
        let realKernel = String(metal[start.lowerBound..<end])

        for fragment in [
            "device const float* a",
            "device const float* b",
            "device float* c",
            "thread_position_in_grid",
            "c[id] = a[id] + b[id]",
        ] {
            XCTAssertTrue(
                realKernel.contains(fragment),
                "kernels.metal vector_add changed: `\(fragment)` is gone, update the README sample"
            )
            XCTAssertTrue(
                readme.contains(fragment),
                "README Metal sample does not match kernels.metal: missing `\(fragment)`"
            )
        }
    }

    /// Claims that were false when this test was written.
    func testKnownFalseClaimsStayAbsent() {
        let readme = self.readme()

        let forbidden: [(claim: String, why: String)] = [
            ("measures GPU utilization", "the demo does not measure GPU utilization"),
            ("measured utilization", "the demo does not measure GPU utilization"),
            ("utilization is", "the demo does not measure GPU utilization"),
            ("speedup of", "unmeasured speedup claim"),
            ("10x faster", "unmeasured speedup claim"),
            ("100x faster", "unmeasured speedup claim"),
            ("iOS target", "Package.swift declares no iOS target"),
            ("iOS 15", "Package.swift declares no iOS platform"),
            ("iOS and", "Package.swift declares no iOS platform"),
            ("automatic translation", "the translation is done by hand"),
            ("automated translator", "the translation is done by hand"),
            ("framework", "this package is an executable demo, not a framework"),
        ]

        for entry in forbidden {
            XCTAssertFalse(
                readme.contains(entry.claim),
                "README contains `\(entry.claim)`, which is false: \(entry.why)"
            )
        }
    }

    /// Platform claims must match Package.swift.
    func testPlatformClaimMatchesPackageManifest() {
        let manifest = source("Package.swift")
        let readme = self.readme()

        if manifest.contains(".macOS(.v12)") {
            XCTAssertTrue(
                readme.contains("macOS 12"),
                "Package.swift requires macOS 12 but the README does not say so"
            )
        }
        XCTAssertFalse(
            manifest.contains(".iOS("),
            "Package.swift now declares an iOS platform; update the README"
        )
    }

    /// Package and target names in the README must match the manifest.
    func testPackageNameMatchesManifest() {
        let manifest = source("Package.swift")
        let readme = self.readme()

        XCTAssertTrue(
            manifest.contains("name: \"MetalKernels\""),
            "the package was renamed; update this test"
        )
        XCTAssertTrue(
            readme.contains("MetalKernels"),
            "README does not name the package"
        )
        XCTAssertFalse(
            readme.contains("GPUComm"),
            "README mentions a GPUComm package that does not exist in this repository"
        )
    }

    /// Referenced repository files must exist.
    func testReferencedFilesExist() {
        let readme = self.readme()
        let pattern = try! NSRegularExpression(pattern: "(Sources|docs)/[A-Za-z0-9_./]+[.](swift|metal|md)")
        let ns = readme as NSString
        let matches = pattern.matches(in: readme, range: NSRange(location: 0, length: ns.length))

        XCTAssertGreaterThan(matches.count, 0, "README references no source paths; the pattern is broken")

        var seen = Set<String>()
        for match in matches {
            let path = ns.substring(with: match.range)
            guard seen.insert(path).inserted else { continue }
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: repoRoot().appendingPathComponent(path).path),
                "README references \(path), which does not exist"
            )
        }
    }

    /// Relative Markdown links must resolve.
    func testRelativeLinksResolve() {
        let readme = self.readme()
        let pattern = try! NSRegularExpression(pattern: "\\]\\(([A-Za-z0-9_./-]+\\.md)\\)")
        let ns = readme as NSString
        let matches = pattern.matches(in: readme, range: NSRange(location: 0, length: ns.length))

        XCTAssertGreaterThan(matches.count, 0, "README has no relative Markdown links to check")

        for match in matches {
            guard let range = Range(match.range(at: 1), in: readme) else { continue }
            let target = String(readme[range])
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: repoRoot().appendingPathComponent(target).path),
                "README links to \(target), which does not exist"
            )
        }
    }

    /// Any timing the README quotes must be labelled as measured somewhere.
    func testTimingClaimsAreMarkedAsMeasured() {
        let readme = self.readme()
        let hasTiming = readme.contains("ms") || readme.contains("GiB/s") || readme.contains("x faster")

        if hasTiming {
            XCTAssertTrue(
                readme.contains("## Measured"),
                "README quotes timings but has no `## Measured ...` heading"
            )
            XCTAssertTrue(
                readme.contains("M1") || readme.contains("Apple") || readme.contains("hardware-dependent"),
                "README quotes timings without naming the hardware"
            )
            XCTAssertTrue(
                readme.contains("will differ") || readme.contains("vary"),
                "README quotes timings without saying they vary by machine"
            )
        }
    }
}