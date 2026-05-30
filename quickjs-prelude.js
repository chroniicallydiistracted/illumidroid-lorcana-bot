// QuickJS host prelude (defines globals the bundle's UMD wrappers expect).
var __g = (typeof globalThis !== "undefined") ? globalThis : this;
__g.process = __g.process || { env: {}, browser: false, version: "v0.0.0", platform: "wasm",
  nextTick: function (f) { Promise.resolve().then(f); }, argv: [], cwd: function () { return "/"; } };
__g.__require = __g.__require || function () { return {}; };
__g.structuredClone = __g.structuredClone || function (v) { return JSON.parse(JSON.stringify(v)); };
__g.performance = __g.performance || { now: function () { return 0; } };
__g.setTimeout = __g.setTimeout || function (f) { Promise.resolve().then(f); return 0; };
__g.clearTimeout = __g.clearTimeout || function () {};
__g.queueMicrotask = __g.queueMicrotask || function (f) { Promise.resolve().then(f); };
