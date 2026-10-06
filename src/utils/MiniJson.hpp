#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace pathways {

struct JsonVal {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool bVal = false;
    double numVal = 0.0;
    std::string strVal;
    std::vector<JsonVal> arr;
    std::unordered_map<std::string, JsonVal> obj;

    static inline const std::string s_emptyString = "";

    bool is_null() const { return type == Null; }
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

    const JsonVal& operator[](const std::string& key) const {
        return at(key);
    }

    const JsonVal& operator[](size_t idx) const {
        static const JsonVal kNull;
        if (type != Array || idx >= arr.size()) return kNull;
        return arr[idx];
    }

    double as_double(double def = 0.0) const { return type == Number ? numVal : def; }
    float as_float(float def = 0.0f) const { return type == Number ? static_cast<float>(numVal) : def; }
    int as_int(int def = 0) const { return type == Number ? static_cast<int>(numVal) : def; }
    uint32_t as_uint(uint32_t def = 0) const { return type == Number ? static_cast<uint32_t>(numVal) : def; }
    bool as_bool(bool def = false) const { return type == Bool ? bVal : def; }
    const std::string& as_string(const std::string& def = s_emptyString) const { return type == String ? strVal : def; }
};

class MiniJsonParser {
    std::string_view s;
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
            v.numVal = std::stod(std::string(s.substr(start, p - start)));
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
    explicit MiniJsonParser(std::string_view input) : s(input) {}

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

} // namespace pathways
