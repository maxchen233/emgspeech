#pragma once
#include "tensor.h"
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstdlib>

// Minimal JSON value (enough to parse model.json)
struct JsonValue {
    enum Type { NUL, NUM, STR, BOOL, ARR, OBJ };
    Type type = NUL;
    double num = 0;
    std::string str;
    bool boolean = false;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    int as_int() const { return (int)num; }
    float as_float() const { return (float)num; }
};

// Very simple JSON parser (handles our model.json format)
class JsonParser {
    const std::string& s;
    size_t pos;

    void skip_ws() {
        while (pos < s.size() && (s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]=='\t')) pos++;
    }
    std::string parse_string() {
        assert(s[pos]=='"'); pos++;
        std::string r;
        while (pos < s.size() && s[pos]!='"') {
            if (s[pos]=='\\') { pos++; r += s[pos]; }
            else r += s[pos];
            pos++;
        }
        pos++; // closing "
        return r;
    }
    JsonValue parse_value() {
        skip_ws();
        JsonValue v;
        if (s[pos]=='"') { v.type = JsonValue::STR; v.str = parse_string(); }
        else if (s[pos]=='{') { v.type = JsonValue::OBJ; v.obj = parse_object(); }
        else if (s[pos]=='[') { v.type = JsonValue::ARR; v.arr = parse_array(); }
        else if (s[pos]=='t') { v.type = JsonValue::BOOL; v.boolean = true; pos+=4; }
        else if (s[pos]=='f') { v.type = JsonValue::BOOL; v.boolean = false; pos+=5; }
        else if (s[pos]=='n') { v.type = JsonValue::NUL; pos+=4; }
        else { // number
            v.type = JsonValue::NUM;
            size_t start = pos;
            if (s[pos]=='-') pos++;
            while (pos < s.size() && (isdigit(s[pos])||s[pos]=='.'||s[pos]=='e'||s[pos]=='E'||s[pos]=='+'||s[pos]=='-')) pos++;
            v.num = std::stod(s.substr(start, pos-start));
        }
        skip_ws();
        return v;
    }
    std::vector<JsonValue> parse_array() {
        assert(s[pos]=='['); pos++; skip_ws();
        std::vector<JsonValue> arr;
        if (s[pos]==']') { pos++; return arr; }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (s[pos]==',') { pos++; skip_ws(); }
            else break;
        }
        assert(s[pos]==']'); pos++;
        return arr;
    }
    std::map<std::string, JsonValue> parse_object() {
        assert(s[pos]=='{'); pos++; skip_ws();
        std::map<std::string, JsonValue> obj;
        if (s[pos]=='}') { pos++; return obj; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws(); assert(s[pos]==':'); pos++;
            obj[key] = parse_value();
            skip_ws();
            if (s[pos]==',') { pos++; }
            else break;
        }
        skip_ws(); assert(s[pos]=='}'); pos++;
        return obj;
    }
public:
    JsonParser(const std::string& s_) : s(s_), pos(0) {}
    JsonValue parse() { return parse_value(); }
};

// Weight loader: reads model.json + .bin files
struct WeightMap {
    std::string base_dir;
    JsonValue manifest;
    std::map<std::string, Tensor> cache;

    void load_manifest(const std::string& dir) {
        base_dir = dir;
        std::string json_path = dir + "/model.json";
        std::ifstream f(json_path);
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open " << json_path << std::endl;
            std::exit(1);
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        JsonParser parser(content);
        manifest = parser.parse();
        std::cerr << "Loaded manifest: " << manifest.obj["tensors"].obj.size()
                  << " tensor entries" << std::endl;
    }

    Tensor get(const std::string& name) {
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;

        auto& tensors = manifest.obj["tensors"].obj;
        auto ti = tensors.find(name);
        if (ti == tensors.end()) {
            std::cerr << "ERROR: tensor '" << name << "' not in manifest" << std::endl;
            std::exit(1);
        }

        auto& info = ti->second;
        if (info.type == JsonValue::BOOL || info.type == JsonValue::NUM) {
            // scalar metadata, not a tensor file
            Tensor t;
            t.shape = {1};
            t.data = { info.type == JsonValue::BOOL ? (info.boolean ? 1.0f : 0.0f) : info.as_float() };
            cache[name] = t;
            return t;
        }

        std::string fname = info.obj["file"].str;
        std::vector<int> shape;
        for (auto& v : info.obj["shape"].arr) shape.push_back(v.as_int());
        int numel = info.obj["numel"].as_int();

        std::string fpath = base_dir + "/" + fname;
        std::ifstream bf(fpath, std::ios::binary);
        if (!bf.is_open()) {
            std::cerr << "ERROR: Cannot open " << fpath << std::endl;
            std::exit(1);
        }

        Tensor t;
        t.shape = shape;
        t.data.resize(numel);
        bf.read(reinterpret_cast<char*>(t.data.data()), numel * sizeof(float));
        cache[name] = t;
        return t;
    }

    int get_int(const std::string& name) {
        return manifest.obj[name].as_int();
    }

    bool get_bool(const std::string& name) {
        auto& tensors = manifest.obj["tensors"].obj;
        auto ti = tensors.find(name);
        if (ti == tensors.end()) return false;
        return ti->second.boolean;
    }
};
