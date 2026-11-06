import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const url =
    process.env.DSSDL_SITE_URL || "https://paoshou-china.github.io/DSSDL/";
assert.match(url, /^https?:\/\//, "Only HTTP(S) test deployments are accepted");
const browser = await chromium.launch();
const errors = [];
await mkdir("site-test-results", { recursive: true });
try {
    for (const [name, width, height] of [
        ["desktop", 1440, 1000],
        ["mobile", 390, 844],
    ]) {
        const page = await browser.newPage({
            viewport: { width, height },
            reducedMotion: "reduce",
        });
        page.on("pageerror", (error) => errors.push(error.message));
        const response = await page.goto(url, { waitUntil: "networkidle" });
        assert.equal(response.status(), 200);
        await page.getByRole("heading", { level: 1 }).waitFor();
        const layout = await page.evaluate(() => ({
            overflow: document.documentElement.scrollWidth > innerWidth + 1,
            broken: [...document.images]
                .filter((i) => !i.complete || !i.naturalWidth)
                .map((i) => i.src),
            colored: [
                ...document
                    .querySelector("canvas")
                    .getContext("2d")
                    .getImageData(
                        0,
                        0,
                        document.querySelector("canvas").width,
                        document.querySelector("canvas").height,
                    ).data,
            ].filter((v, i) => i % 4 === 3 && v > 0).length,
        }));
        assert.equal(layout.overflow, false, `${name}: horizontal overflow`);
        assert.deepEqual(layout.broken, []);
        assert.ok(layout.colored > 1000, "Plot must contain visible pixels");
        await page.screenshot({
            path: `site-test-results/${name}.png`,
            fullPage: true,
        });
        await page.getByRole("button", { name: "FHN", exact: true }).click();
        assert.match(await page.locator("#plot-title").textContent(), /FHN/);
        await page.getByRole("button", { name: "核验", exact: true }).click();
        assert.match(
            await page.locator("#command-text").textContent(),
            /dssdl verify/,
        );
        await page
            .getByRole("button", { name: "播放轨迹", exact: true })
            .click();
        await page
            .getByRole("button", { name: "暂停轨迹", exact: true })
            .waitFor();
        await page
            .getByRole("button", { name: "暂停轨迹", exact: true })
            .click();
        const metricValues = await page
            .locator(".metric-value")
            .allTextContents();
        const command = await page.locator("#command-text").textContent();
        await page.getByRole("button", { name: "EN", exact: true }).click();
        assert.equal(await page.locator("html").getAttribute("lang"), "en");
        assert.match(
            await page.locator("#plot-title").textContent(),
            /FHN.*Fixed-parameter/,
        );
        assert.deepEqual(
            await page.locator(".metric-value").allTextContents(),
            metricValues,
        );
        assert.equal(
            await page.locator("#command-text").textContent(),
            command,
        );
        assert.equal(
            await page
                .getByRole("button", { name: "Copy commands", exact: true })
                .isVisible(),
            true,
        );
        assert.equal(
            await page.evaluate(
                () => document.documentElement.scrollWidth > innerWidth + 1,
            ),
            false,
        );
        await page.screenshot({
            path: `site-test-results/${name}-en.png`,
            fullPage: true,
        });
        await page.reload({ waitUntil: "networkidle" });
        assert.equal(await page.locator("html").getAttribute("lang"), "en");
        await page.getByRole("button", { name: "中文", exact: true }).click();
        assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
        assert.deepEqual(errors, []);
        await page.close();
        console.log(
            `${name}: bilingual layout, assets, plot pixels, controls and language persistence passed`,
        );
    }
} finally {
    await browser.close();
}
