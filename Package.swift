// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalKernels",
    platforms: [
        .macOS(.v12)
    ],
    dependencies: [],
    targets: [
        // Documentation drift guard. Reads README.md and the demo source as
        // text; it does not create a Metal device, so it runs on CI runners
        // without a GPU.
        .testTarget(
            name: "MetalKernelsTests",
            dependencies: ["MetalKernels"],
            path: "Tests/MetalKernelsTests"
        ),
        .executableTarget(
            name: "MetalKernels",
            dependencies: [],
            resources: [
                .copy("kernels.metal")
            ]
        )
    ]
)
