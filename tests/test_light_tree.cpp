#include "scene/LightTree.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <unordered_set>

using namespace pathways;

void assert_near(float a, float b, float eps = 0.01f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Hierarchical Light Tree (FEAT-02)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Empty light list
    {
        std::vector<LightGPU> lights;
        std::vector<LightTreeNodeGPU> nodes;
        buildLightTree(lights, nodes);
        check_true(nodes.size() == 1, "Empty tree should have 1 dummy node");
        check_true(nodes[0].children.z == 1u, "Dummy node is marked leaf");
        std::cout << "[PASS] Empty tree initialization verified." << std::endl;
    }

    // 2. Single light
    {
        std::vector<LightGPU> lights;
        LightGPU l{};
        l.position = glm::vec4(1.0f, 2.0f, 3.0f, static_cast<float>(LIGHT_AREA_QUAD));
        l.u = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        l.v = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
        l.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        l.emission = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f); // area = 1.0
        lights.push_back(l);

        std::vector<LightTreeNodeGPU> nodes;
        buildLightTree(lights, nodes);
        check_true(nodes.size() == 1, "Single light tree has 1 node");
        check_true(nodes[0].children.z == 1u, "Single node is leaf");
        check_true(nodes[0].children.x == 0u, "Leaf light index is 0");
        float expectedFlux = calculateLightFlux(l);
        assert_near(nodes[0].bboxMin.w, expectedFlux, 0.01f, "Single light flux matches");
        std::cout << "[PASS] Single light tree verified." << std::endl;
    }

    // 3. Multi-light hierarchy (64 lights across 3D space)
    {
        std::vector<LightGPU> lights;
        float totalExpectedFlux = 0.0f;
        for (int i = 0; i < 64; ++i) {
            LightGPU l{};
            float x = static_cast<float>(i % 4) * 5.0f;
            float y = static_cast<float>((i / 4) % 4) * 5.0f;
            float z = static_cast<float>(i / 16) * 5.0f;

            if (i % 3 == 0) {
                l.position = glm::vec4(x, y, z, static_cast<float>(LIGHT_AREA_QUAD));
                l.u = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                l.v = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
                l.normal = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
                l.emission = glm::vec4(5.0f + static_cast<float>(i), 5.0f, 5.0f, 1.0f);
            } else if (i % 3 == 1) {
                l.position = glm::vec4(x, y, z, static_cast<float>(LIGHT_SPOT));
                l.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
                l.u = glm::vec4(0.0f, 0.0f, 0.0f, 0.8f); // inner cos
                l.v = glm::vec4(0.0f, 0.0f, 0.0f, 0.6f); // outer cos
                l.emission = glm::vec4(20.0f, 20.0f, 20.0f, 1.0f);
            } else {
                l.position = glm::vec4(x, y, z, static_cast<float>(LIGHT_DIRECTIONAL));
                l.normal = glm::vec4(0.577f, 0.577f, 0.577f, 0.0f);
                l.emission = glm::vec4(15.0f, 15.0f, 15.0f, 1.0f);
            }
            lights.push_back(l);
            totalExpectedFlux += calculateLightFlux(l);
        }

        std::vector<LightTreeNodeGPU> nodes;
        buildLightTree(lights, nodes);

        check_true(!nodes.empty(), "Nodes vector not empty");
        // Root is at index 0
        assert_near(nodes[0].bboxMin.w, totalExpectedFlux, 0.1f, "Root total radiant flux matches");

        // Verify leaf nodes and light coverage
        std::unordered_set<uint32_t> coveredLightIndices;
        int leafCount = 0;
        int maxDepth = 0;

        auto traverse = [&](auto& self, uint32_t nodeIdx, int depth) -> void {
            if (depth > maxDepth) maxDepth = depth;
            check_true(nodeIdx < nodes.size(), "Valid node index");
            const auto& node = nodes[nodeIdx];

            if (node.children.z == 1u) {
                // Leaf
                leafCount++;
                uint32_t lIdx = node.children.x;
                check_true(lIdx < lights.size(), "Valid leaf light index");
                check_true(coveredLightIndices.find(lIdx) == coveredLightIndices.end(), "Each light visited exactly once");
                coveredLightIndices.insert(lIdx);
            } else {
                // Internal node
                uint32_t left = node.children.x;
                uint32_t right = node.children.y;
                check_true(left < nodes.size() && right < nodes.size(), "Children indices valid");

                // Verify bounding box bounds children
                const auto& lNode = nodes[left];
                const auto& rNode = nodes[right];
                check_true(node.bboxMin.x <= lNode.bboxMin.x + 1e-4f && node.bboxMin.y <= lNode.bboxMin.y + 1e-4f && node.bboxMin.z <= lNode.bboxMin.z + 1e-4f, "Bounding box bounds left min");
                check_true(node.bboxMax.x >= lNode.bboxMax.x - 1e-4f && node.bboxMax.y >= lNode.bboxMax.y - 1e-4f && node.bboxMax.z >= lNode.bboxMax.z - 1e-4f, "Bounding box bounds left max");
                check_true(node.bboxMin.x <= rNode.bboxMin.x + 1e-4f && node.bboxMin.y <= rNode.bboxMin.y + 1e-4f && node.bboxMin.z <= rNode.bboxMin.z + 1e-4f, "Bounding box bounds right min");
                check_true(node.bboxMax.x >= rNode.bboxMax.x - 1e-4f && node.bboxMax.y >= rNode.bboxMax.y - 1e-4f && node.bboxMax.z >= rNode.bboxMax.z - 1e-4f, "Bounding box bounds right max");

                // Verify flux summation
                assert_near(node.bboxMin.w, lNode.bboxMin.w + rNode.bboxMin.w, 0.01f, "Flux conservation in internal node");

                self(self, left, depth + 1);
                self(self, right, depth + 1);
            }
        };

        traverse(traverse, 0u, 1);

        check_true(leafCount == 64, "All 64 leaves present");
        check_true(coveredLightIndices.size() == 64, "All 64 lights covered");
        check_true(maxDepth <= 10, "Tree depth is O(log2 N)");
        std::cout << "[PASS] 64-light tree hierarchy, flux conservation, and bounds verified (depth: " << maxDepth << ")." << std::endl;
    }

    std::cout << "\nALL HIERARCHICAL LIGHT TREE TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
