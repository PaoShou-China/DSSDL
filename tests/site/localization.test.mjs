import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const context = { window: {} };
vm.runInNewContext(readFileSync("docs/i18n.js", "utf8"), context);
const translations = context.window.DSSDL_TRANSLATIONS;

test("every website message has Chinese and English text", () => {
    assert.ok(Object.keys(translations).length >= 100);
    for (const [key, values] of Object.entries(translations)) {
        assert.equal(values.length, 2, key);
        for (const value of values) {
            assert.equal(typeof value, "string", key);
            assert.ok(value.trim().length > 0, key);
            assert.ok(
                !value.includes("<"),
                `${key}: translations must be text`,
            );
        }
        if (key !== "language") {
            assert.doesNotMatch(values[1], /[\u3400-\u9fff]/, key);
        }
    }
});

test("all static and dynamic translation references exist", () => {
    const html = readFileSync("docs/index.html", "utf8");
    const app = readFileSync("docs/app.js", "utf8");
    const keys = [
        ...html.matchAll(/data-i18n(?:-aria-label|-title|-alt)?="([^"]+)"/g),
        ...app.matchAll(/translate\("([^"]+)"\)/g),
    ].map((match) => match[1]);
    for (const key of [
        ...keys,
        "play",
        "pause",
        "copied",
        "copyFailed",
        "lorenzContext",
        "fhnContext",
        "installNote",
        "verifyNote",
        "runNote",
        "openMenu",
        "closeMenu",
    ])
        assert.ok(translations[key], `Missing translation: ${key}`);
    assert.match(html, /data-language="zh"/);
    assert.match(html, /data-language="en"/);
});
