// Runs the race view's app.js in Node against a small stand-in DOM, then the async function
// body read from stdin, and prints what that body returns as JSON. Used by test_web.py.
//
// Usage: node web_harness.cjs path/to/app.js < snippet.js

"use strict";

const fs = require("fs");
const vm = require("vm");

class FakeElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];  // FakeElement or plain strings (text nodes)
    this.ownText = "";
    this.className = "";
    this.title = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    const self = this;
    this.classList = {
      contains: (c) => self.className.split(" ").includes(c),
      add: (c) => { if (!self.classList.contains(c)) self.className = `${self.className} ${c}`.trim(); },
      remove: (c) => { self.className = self.className.split(" ").filter((x) => x && x !== c).join(" "); },
      toggle: (c, on) => (on ? self.classList.add(c) : self.classList.remove(c)),
    };
  }

  append(...nodes) { this.children.push(...nodes); }

  replaceChildren(...nodes) {
    this.children = [];
    this.ownText = "";
    this.append(...nodes);
  }

  set textContent(value) {
    this.children = [];
    this.ownText = String(value);
  }

  get textContent() {
    return this.ownText + this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }

  // Only ".class" selectors, searched depth first.
  querySelector(selector) {
    for (const c of this.children) {
      if (typeof c === "string") continue;
      if (c.classList.contains(selector.slice(1))) return c;
      const inner = c.querySelector(selector);
      if (inner) return inner;
    }
    return null;
  }

  addEventListener() {}
}

const elements = new Map();
const document = {
  getElementById: (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement("div"));
    return elements.get(id);
  },
  createElement: (tag) => new FakeElement(tag),
};

// Each row of an element: its class and its pieces as [text, class, title].
const dump = (id) => document.getElementById(id).children.map((r) => ({
  cls: r.className,
  pieces: r.children.map((c) => (typeof c === "string" ? [c, "", ""] : [c.textContent, c.className, c.title])),
}));
const text = (id) => document.getElementById(id).textContent;
const cls = (id) => document.getElementById(id).className;

const context = vm.createContext({
  document,
  window: { matchMedia: () => ({ matches: true }) },
  fetch: () => new Promise(() => {}),  // the page's start-up requests never answer
  performance,
  requestAnimationFrame: () => 0,
  TextDecoder,
  TextEncoder,
  console,
  dump,
  text,
  cls,
});

vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context, { filename: "app.js" });
const snippet = fs.readFileSync(0, "utf8");
vm.runInContext(`(async () => {\n${snippet}\n})()`, context).then(
  (result) => process.stdout.write(JSON.stringify(result)),
  (err) => {
    process.stderr.write(String(err && err.stack ? err.stack : err));
    process.exitCode = 1;
  },
);
