#include "scene/CameraPath.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {

struct JsonVal {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool bVal = false;
    double numVal = 0.0;
    std::string strVal;
    std::vector<JsonVal> arr;
    std::unordered_map<std::string, JsonVal> obj;

    bool is_object() const { return type == Object; }
    bool is_array() const { return type == Array; }
    bool is_string() const { return type == String; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }

    bool contains(const std::string& key) const {
        return type == Object && obj.find(key) != obj.end();
    }

    const JsonVal& at(const std::string& key) const {
        static const JsonVal kNull;
        if (type != Object) return kNull;
        auto it = obj.find(key);
        return it != obj.end() ? it->second : kNull;
    }

    const JsonVal& operator[](size_t idx) const {
        static const JsonVal kNull;
        if (type != Array || idx >= arr.size()) return kNull;
        return arr[idx];
    }
};

class MiniJsonParser {
    const std::string& s;
    size_t p = 0;

    void skipWs() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) {
            p++;
        }
    }

    char peek() { skipWs(); return p < s.size() ? s[p] : '\0'; }
    char get() { skipWs(); return p < s.size() ? s[p++] : '\0'; }

    std::string parseStr() {
        if (get() != '"') return "";
        std::string res;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return res;
            if (c == '\\' && p < s.size()) {
                char esc = s[p++];
                if (esc == 'n') res += '\n';
                else if (esc == 't') res += '\t';
                else if (esc == 'r') res += '\r';
                else if (esc == '"' || esc == '\\' || esc == '/') res += esc;
                else res += esc;
            } else {
                res += c;
            }
        }
        return res;
    }

    JsonVal parseNum() {
        skipWs();
        size_t start = p;
        if (p < s.size() && (s[p] == '-' || s[p] == '+')) p++;
        while (p < s.size() && ((s[p] >= '0' && s[p] <= '9') || s[p] == '.' || s[p] == 'e' || s[p] == 'E' || s[p] == '-' || s[p] == '+')) {
            p++;
        }
        JsonVal v;
        v.type = JsonVal::Number;
        try {
            v.numVal = std::stod(s.substr(start, p - start));
        } catch (...) {
            v.numVal = 0.0;
        }
        return v;
    }

    JsonVal parseObj() {
        JsonVal v;
        v.type = JsonVal::Object;
        if (get() != '{') return v;
        while (p < s.size()) {
            char c = peek();
            if (c == '}') { get(); break; }
            if (c == ',') { get(); continue; }
            std::string key = parseStr();
            skipWs();
            if (get() != ':') break;
            v.obj[key] = parseVal();
        }
        return v;
    }

    JsonVal parseArr() {
        JsonVal v;
        v.type = JsonVal::Array;
        if (get() != '[') return v;
        while (p < s.size()) {
            char c = peek();
            if (c == ']') { get(); break; }
            if (c == ',') { get(); continue; }
            v.arr.push_back(parseVal());
        }
        return v;
    }

public:
    MiniJsonParser(const std::string& input) : s(input) {}

    JsonVal parseVal() {
        skipWs();
        char c = peek();
        if (c == '{') return parseObj();
        if (c == '[') return parseArr();
        if (c == '"') {
            JsonVal v;
            v.type = JsonVal::String;
            v.strVal = parseStr();
            return v;
        }
        if (c == 't' || c == 'f') {
            JsonVal v;
            v.type = JsonVal::Bool;
            if (s.compare(p, 4, "true") == 0) { p += 4; v.bVal = true; }
            else if (s.compare(p, 5, "false") == 0) { p += 5; v.bVal = false; }
            return v;
        }
        if (c == 'n') {
            if (s.compare(p, 4, "null") == 0) p += 4;
            return JsonVal();
        }
        if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
            return parseNum();
        }
        return JsonVal();
    }
};

} // namespace

namespace pathways {

CameraPath::CameraPath(const std::string& name, PathInterpolation mode)
    : m_name(name), m_interpolation(mode) {}

void CameraPath::addKeyframe(const CameraKeyframe& kf) {
    if (!m_keyframes.empty() && kf.time < m_keyframes.back().time) {
        Logger::warn("CameraPath [{}]: Added keyframe with non-monotonic time ({} < {}). Sorting keyframes.",
                     m_name, kf.time, m_keyframes.back().time);
    }
    m_keyframes.push_back(kf);
    std::sort(m_keyframes.begin(), m_keyframes.end(), [](const CameraKeyframe& a, const CameraKeyframe& b) {
        return a.time < b.time;
    });
}

float CameraPath::getStartTime() const {
    if (m_keyframes.empty()) return 0.0f;
    return m_keyframes.front().time;
}

float CameraPath::getEndTime() const {
    if (m_keyframes.empty()) return 0.0f;
    return m_keyframes.back().time;
}

float CameraPath::getDuration() const {
    if (m_keyframes.size() < 2) return 0.0f;
    return getEndTime() - getStartTime();
}

glm::vec3 CameraPath::evaluateCentripetalCatmullRom(
    const glm::vec3& p0, const glm::vec3& p1,
    const glm::vec3& p2, const glm::vec3& p3,
    float t0, float t1, float t2, float t3,
    float t, float alpha) {

    // Epsilon to prevent division by zero in degenerate / stationary points
    constexpr float eps = 1e-6f;

    float dt01 = std::pow(std::max(glm::length(p1 - p0), eps), alpha);
    float dt12 = std::pow(std::max(glm::length(p2 - p1), eps), alpha);
    float dt23 = std::pow(std::max(glm::length(p3 - p2), eps), alpha);

    float knot0 = 0.0f;
    float knot1 = knot0 + dt01;
    float knot2 = knot1 + dt12;
    float knot3 = knot2 + dt23;

    // Rescale evaluation time into knot parameter space
    float u = std::clamp((t - t1) / std::max(t2 - t1, eps), 0.0f, 1.0f);
    float tau = knot1 + u * (knot2 - knot1);

    // Barry-Goldman pyramidal formulation
    auto safeDiv = [](float num, float den) -> float {
        return (std::abs(den) > 1e-6f) ? (num / den) : 0.0f;
    };

    glm::vec3 A1 = safeDiv(knot1 - tau, knot1 - knot0) * p0 + safeDiv(tau - knot0, knot1 - knot0) * p1;
    glm::vec3 A2 = safeDiv(knot2 - tau, knot2 - knot1) * p1 + safeDiv(tau - knot1, knot2 - knot1) * p2;
    glm::vec3 A3 = safeDiv(knot3 - tau, knot3 - knot2) * p2 + safeDiv(tau - knot2, knot3 - knot2) * p3;

    glm::vec3 B1 = safeDiv(knot2 - tau, knot2 - knot0) * A1 + safeDiv(tau - knot0, knot2 - knot0) * A2;
    glm::vec3 B2 = safeDiv(knot3 - tau, knot3 - knot1) * A2 + safeDiv(tau - knot1, knot3 - knot1) * A3;

    glm::vec3 C = safeDiv(knot2 - tau, knot2 - knot1) * B1 + safeDiv(tau - knot1, knot2 - knot1) * B2;
    return C;
}

glm::vec3 CameraPath::evaluateCubicBezier(
    const glm::vec3& p0, const glm::vec3& c0,
    const glm::vec3& c1, const glm::vec3& p1,
    float u) {
    u = std::clamp(u, 0.0f, 1.0f);
    float oneMinusU = 1.0f - u;
    float oneMinusU2 = oneMinusU * oneMinusU;
    float oneMinusU3 = oneMinusU2 * oneMinusU;
    float u2 = u * u;
    float u3 = u2 * u;

    return oneMinusU3 * p0 +
           3.0f * oneMinusU2 * u * c0 +
           3.0f * oneMinusU * u2 * c1 +
           u3 * p1;
}

glm::vec3 CameraPath::evaluateCubicBezierDerivative(
    const glm::vec3& p0, const glm::vec3& c0,
    const glm::vec3& c1, const glm::vec3& p1,
    float u) {
    u = std::clamp(u, 0.0f, 1.0f);
    float oneMinusU = 1.0f - u;
    float oneMinusU2 = oneMinusU * oneMinusU;
    float u2 = u * u;

    return 3.0f * oneMinusU2 * (c0 - p0) +
           6.0f * oneMinusU * u * (c1 - c0) +
           3.0f * u2 * (p1 - c1);
}

CameraSample CameraPath::evaluate(float timeSeconds, bool loopOverride) const {
    CameraSample sample{};
    if (m_keyframes.empty()) {
        sample.isStationary = true;
        return sample;
    }

    if (m_keyframes.size() == 1) {
        const auto& kf = m_keyframes.front();
        sample.position = kf.position;
        sample.target = kf.target;
        sample.up = kf.up;
        sample.fov = kf.fov;
        sample.velocity = glm::vec3(0.0f);
        sample.speed = 0.0f;
        sample.isStationary = true;
        return sample;
    }

    bool loop = m_loop || loopOverride;
    float startTime = getStartTime();
    float endTime = getEndTime();
    float duration = endTime - startTime;

    float t = timeSeconds;
    if (loop && duration > 1e-4f) {
        float offset = std::fmod(t - startTime, duration);
        if (offset < 0.0f) offset += duration;
        t = startTime + offset;
    } else {
        t = std::clamp(t, startTime, endTime);
    }

    // Binary search for keyframe interval [idx, idx+1] such that kf[idx].time <= t <= kf[idx+1].time
    auto it = std::upper_bound(m_keyframes.begin(), m_keyframes.end(), t,
        [](float val, const CameraKeyframe& kf) {
            return val < kf.time;
        });

    size_t idx = 0;
    if (it == m_keyframes.begin()) {
        idx = 0;
    } else if (it == m_keyframes.end()) {
        idx = m_keyframes.size() - 2;
    } else {
        idx = std::distance(m_keyframes.begin(), it) - 1;
    }

    const auto& kfA = m_keyframes[idx];
    const auto& kfB = m_keyframes[idx + 1];

    float segDt = kfB.time - kfA.time;
    float u = (segDt > 1e-6f) ? std::clamp((t - kfA.time) / segDt, 0.0f, 1.0f) : 0.0f;

    // Evaluate position, target, and velocity based on active interpolation mode
    if (m_interpolation == PathInterpolation::Linear) {
        sample.position = glm::mix(kfA.position, kfB.position, u);
        sample.target = glm::mix(kfA.target, kfB.target, u);
        sample.velocity = (segDt > 1e-6f) ? ((kfB.position - kfA.position) / segDt) : glm::vec3(0.0f);
    }
    else if (m_interpolation == PathInterpolation::CatmullRom) {
        // Collect 4 points for Catmull-Rom
        glm::vec3 p0, p1, p2, p3;
        glm::vec3 t0_pt, t1_pt, t2_pt, t3_pt;
        float time0, time1, time2, time3;

        p1 = kfA.position;
        p2 = kfB.position;
        t1_pt = kfA.target;
        t2_pt = kfB.target;
        time1 = kfA.time;
        time2 = kfB.time;

        if (idx > 0) {
            p0 = m_keyframes[idx - 1].position;
            t0_pt = m_keyframes[idx - 1].target;
            time0 = m_keyframes[idx - 1].time;
        } else if (loop) {
            p0 = m_keyframes[m_keyframes.size() - 2].position;
            t0_pt = m_keyframes[m_keyframes.size() - 2].target;
            time0 = time1 - (m_keyframes.back().time - m_keyframes[m_keyframes.size() - 2].time);
        } else {
            // Natural boundary reflection
            p0 = p1 - (p2 - p1);
            t0_pt = t1_pt - (t2_pt - t1_pt);
            time0 = time1 - (time2 - time1);
        }

        if (idx + 2 < m_keyframes.size()) {
            p3 = m_keyframes[idx + 2].position;
            t3_pt = m_keyframes[idx + 2].target;
            time3 = m_keyframes[idx + 2].time;
        } else if (loop) {
            p3 = m_keyframes[1].position;
            t3_pt = m_keyframes[1].target;
            time3 = time2 + (m_keyframes[1].time - m_keyframes[0].time);
        } else {
            // Natural boundary reflection
            p3 = p2 + (p2 - p1);
            t3_pt = t2_pt + (t2_pt - t1_pt);
            time3 = time2 + (time2 - time1);
        }

        sample.position = evaluateCentripetalCatmullRom(p0, p1, p2, p3, time0, time1, time2, time3, t);
        sample.target = evaluateCentripetalCatmullRom(t0_pt, t1_pt, t2_pt, t3_pt, time0, time1, time2, time3, t);

        constexpr float deltaT = 0.002f;
        glm::vec3 posPlus = evaluateCentripetalCatmullRom(p0, p1, p2, p3, time0, time1, time2, time3, t + deltaT);
        glm::vec3 posMinus = evaluateCentripetalCatmullRom(p0, p1, p2, p3, time0, time1, time2, time3, t - deltaT);
        sample.velocity = (posPlus - posMinus) / (2.0f * deltaT);
    }
    else if (m_interpolation == PathInterpolation::Bezier) {
        // Compute control handles: use authored tangents if non-zero, otherwise synthesize smooth Catmull-Rom tangents
        glm::vec3 c0 = kfA.position;
        glm::vec3 c1 = kfB.position;

        if (glm::length(kfA.outTangent) > 1e-4f) {
            c0 = kfA.position + kfA.outTangent;
        } else {
            glm::vec3 pPrev = (idx > 0) ? m_keyframes[idx - 1].position : (kfA.position - (kfB.position - kfA.position));
            c0 = kfA.position + (kfB.position - pPrev) * (1.0f / 6.0f);
        }

        if (glm::length(kfB.inTangent) > 1e-4f) {
            c1 = kfB.position + kfB.inTangent;
        } else {
            glm::vec3 pNext = (idx + 2 < m_keyframes.size()) ? m_keyframes[idx + 2].position : (kfB.position + (kfB.position - kfA.position));
            c1 = kfB.position - (pNext - kfA.position) * (1.0f / 6.0f);
        }

        sample.position = evaluateCubicBezier(kfA.position, c0, c1, kfB.position, u);
        sample.velocity = (segDt > 1e-6f) ? (evaluateCubicBezierDerivative(kfA.position, c0, c1, kfB.position, u) / segDt) : glm::vec3(0.0f);

        // Target Bézier
        glm::vec3 tc0 = kfA.target;
        glm::vec3 tc1 = kfB.target;
        if (glm::length(kfA.targetOutTangent) > 1e-4f) {
            tc0 = kfA.target + kfA.targetOutTangent;
        } else {
            glm::vec3 tPrev = (idx > 0) ? m_keyframes[idx - 1].target : (kfA.target - (kfB.target - kfA.target));
            tc0 = kfA.target + (kfB.target - tPrev) * (1.0f / 6.0f);
        }
        if (glm::length(kfB.targetInTangent) > 1e-4f) {
            tc1 = kfB.target + kfB.targetInTangent;
        } else {
            glm::vec3 tNext = (idx + 2 < m_keyframes.size()) ? m_keyframes[idx + 2].target : (kfB.target + (kfB.target - kfA.target));
            tc1 = kfB.target - (tNext - kfA.target) * (1.0f / 6.0f);
        }
        sample.target = evaluateCubicBezier(kfA.target, tc0, tc1, kfB.target, u);
    }

    // Up vector: smooth normalized interpolation
    sample.up = glm::normalize(glm::mix(kfA.up, kfB.up, u));

    // FOV: smooth scalar interpolation
    sample.fov = glm::mix(kfA.fov, kfB.fov, u);

    sample.speed = glm::length(sample.velocity);
    sample.isStationary = (sample.speed < 1e-4f);

    return sample;
}

std::unique_ptr<CameraPath> CameraPath::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        Logger::error("CameraPath: Failed to open file: {}", filepath);
        return nullptr;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return loadFromString(buffer.str());
}

std::unique_ptr<CameraPath> CameraPath::loadFromString(const std::string& jsonString) {
    try {
        MiniJsonParser parser(jsonString);
        JsonVal root = parser.parseVal();
        if (!root.is_object()) {
            Logger::error("CameraPath: Root JSON value is not an object");
            return nullptr;
        }

        const auto& obj = root;
        std::string name = "CameraPath";
        if (obj.contains("name") && obj.at("name").is_string()) {
            name = obj.at("name").strVal;
        }

        PathInterpolation interp = PathInterpolation::CatmullRom;
        if (obj.contains("interpolation") && obj.at("interpolation").is_string()) {
            std::string modeStr = obj.at("interpolation").strVal;
            std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::tolower);
            if (modeStr == "bezier") {
                interp = PathInterpolation::Bezier;
            } else if (modeStr == "linear") {
                interp = PathInterpolation::Linear;
            } else {
                interp = PathInterpolation::CatmullRom;
            }
        }

        auto path = std::make_unique<CameraPath>(name, interp);

        if (obj.contains("loop") && obj.at("loop").is_bool()) {
            path->setLoop(obj.at("loop").bVal);
        }

        if (!obj.contains("keyframes") || !obj.at("keyframes").is_array()) {
            Logger::error("CameraPath: Missing or invalid 'keyframes' array");
            return nullptr;
        }

        const auto& kfArray = obj.at("keyframes").arr;
        for (const auto& elem : kfArray) {
            if (!elem.is_object()) continue;
            const auto& kfObj = elem;

            CameraKeyframe kf{};

            // Time
            if (kfObj.contains("time") && kfObj.at("time").is_number()) {
                kf.time = static_cast<float>(kfObj.at("time").numVal);
            }

            // Position [x, y, z]
            auto parseVec3 = [](const JsonVal& o, const char* key, glm::vec3 defVal) -> glm::vec3 {
                if (!o.contains(key) || !o.at(key).is_array()) return defVal;
                const auto& arr = o.at(key).arr;
                glm::vec3 v = defVal;
                if (arr.size() >= 3) {
                    auto toFloat = [](const JsonVal& val) -> float {
                        if (val.is_number()) return static_cast<float>(val.numVal);
                        return 0.0f;
                    };
                    v.x = toFloat(arr[0]);
                    v.y = toFloat(arr[1]);
                    v.z = toFloat(arr[2]);
                }
                return v;
            };

            kf.position = parseVec3(kfObj, "position", glm::vec3(0.0f));
            if (!kfObj.contains("position") && kfObj.contains("pos")) {
                kf.position = parseVec3(kfObj, "pos", glm::vec3(0.0f));
            }

            // Target [x, y, z]
            kf.target = parseVec3(kfObj, "target", glm::vec3(0.0f, 0.0f, -1.0f));
            if (!kfObj.contains("target") && kfObj.contains("look_at")) {
                kf.target = parseVec3(kfObj, "look_at", glm::vec3(0.0f, 0.0f, -1.0f));
            }

            // Up [x, y, z]
            kf.up = parseVec3(kfObj, "up", glm::vec3(0.0f, 1.0f, 0.0f));

            // FOV
            if (kfObj.contains("fov") && kfObj.at("fov").is_number()) {
                kf.fov = static_cast<float>(kfObj.at("fov").numVal);
            }

            // Tangents
            kf.inTangent = parseVec3(kfObj, "in_tangent", glm::vec3(0.0f));
            kf.outTangent = parseVec3(kfObj, "out_tangent", glm::vec3(0.0f));
            kf.targetInTangent = parseVec3(kfObj, "target_in_tangent", glm::vec3(0.0f));
            kf.targetOutTangent = parseVec3(kfObj, "target_out_tangent", glm::vec3(0.0f));

            path->addKeyframe(kf);
        }

        Logger::info("CameraPath: Loaded '{}' with {} keyframes (Duration: {:.2f}s, Interpolation: {})",
                     path->getName(), path->getKeyframeCount(), path->getDuration(),
                     (interp == PathInterpolation::CatmullRom ? "CatmullRom" : (interp == PathInterpolation::Bezier ? "Bezier" : "Linear")));
        return path;
    } catch (const std::exception& e) {
        Logger::error("CameraPath: JSON parse error: {}", e.what());
        return nullptr;
    }
}

bool CameraPath::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        Logger::error("CameraPath: Failed to open file for writing: {}", filepath);
        return false;
    }
    file << serializeToJSON();
    return true;
}

std::string CameraPath::serializeToJSON() const {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"name\": \"" << m_name << "\",\n";
    ss << "  \"interpolation\": \"" << (m_interpolation == PathInterpolation::CatmullRom ? "catmull_rom" : (m_interpolation == PathInterpolation::Bezier ? "bezier" : "linear")) << "\",\n";
    ss << "  \"loop\": " << (m_loop ? "true" : "false") << ",\n";
    ss << "  \"keyframes\": [\n";

    for (size_t i = 0; i < m_keyframes.size(); ++i) {
        const auto& kf = m_keyframes[i];
        ss << "    {\n";
        ss << "      \"time\": " << kf.time << ",\n";
        ss << "      \"position\": [" << kf.position.x << ", " << kf.position.y << ", " << kf.position.z << "],\n";
        ss << "      \"target\": [" << kf.target.x << ", " << kf.target.y << ", " << kf.target.z << "],\n";
        ss << "      \"up\": [" << kf.up.x << ", " << kf.up.y << ", " << kf.up.z << "],\n";
        ss << "      \"fov\": " << kf.fov;

        if (m_interpolation == PathInterpolation::Bezier) {
            ss << ",\n";
            ss << "      \"in_tangent\": [" << kf.inTangent.x << ", " << kf.inTangent.y << ", " << kf.inTangent.z << "],\n";
            ss << "      \"out_tangent\": [" << kf.outTangent.x << ", " << kf.outTangent.y << ", " << kf.outTangent.z << "]";
        }
        ss << "\n    }" << (i + 1 < m_keyframes.size() ? "," : "") << "\n";
    }

    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
}

} // namespace pathways
