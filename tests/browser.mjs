import { chromium } from "playwright";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
const base = process.env.BASE_URL || "http://127.0.0.1:8353/";
let server;
if (!process.env.BASE_URL) {
  server = spawn(
    "python3",
    ["-m", "http.server", "8353", "--bind", "127.0.0.1", "--directory", "docs"],
    { stdio: "ignore" },
  );
  await new Promise((r) => setTimeout(r, 500));
}
const browser = await chromium.launch({ headless: true });
let count = 0;
const ok = (v, m) => {
  assert.ok(v, m);
  count++;
};
await mkdir("/tmp/tangentlab-shots", { recursive: true });
try {
  for (const width of [1440, 1024, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } }),
      errors = [],
      failed = [],
      third = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    page.on("requestfailed", (r) => failed.push(r.url()));
    page.on("request", (r) => {
      if (!r.url().startsWith(base)) third.push(r.url());
    });
    await page.goto(base);
    await page.waitForFunction(() =>
      document.querySelector("#status").textContent.includes("APPLIED POINT"),
    );
    ok((await page.locator("#greeks tr").count()) === 5, "5Greekrows");
    ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "nooverflow",
    );
    await page.screenshot({ path: `/tmp/tangentlab-shots/hero-${width}.png` });
    const expected = await page.evaluate(async () => {
      const { fit, predict, analytic, evaluate } = await import("./core.js");
      const m = fit();
      return {
        reference: analytic([100, 1, 0.3]),
        predicted: predict(m, [100, 1, 0.3]),
        metrics: evaluate(m).metrics,
        weights: m.weights,
        meta: m,
      };
    });
    const cells = await page
        .locator("#greeks tr td:nth-child(2)")
        .allTextContents(),
      pred = await page.locator("#greeks tr td:nth-child(3)").allTextContents();
    ok(
      cells.every(
        (v, j) =>
          Math.abs(Number(v) - expected.reference.greeks[j]) <= 0.00000051,
      ),
      "actualreference",
    );
    ok(
      pred.every(
        (v, j) => Math.abs(Number(v) - expected.predicted[j]) <= 0.00000051,
      ),
      "actualsurrogate",
    );
    ok(
      Number(await page.locator("#price").innerText()).toFixed(6) ===
        expected.reference.price.toFixed(6),
      "priceactual",
    );
    ok(
      (await page.locator("#model-meta").innerText()) ===
        "343 train rows / 35 features",
      "trainedmetadata",
    );
    ok((await page.locator("#audit-metrics tr").count()) === 10, "10auditrows");
    const mae = await page
      .locator("#audit-metrics tr td:nth-child(2)")
      .allTextContents();
    ok(
      mae.every(
        (v, i) => Math.abs(Number(v) - expected.metrics[i].mae) < 0.00000051,
      ),
      "actualaudit",
    );
    ok((await page.locator("#holdout tr").count()) === 216, "allholdout");
    const pixels = await page.locator("#curve").evaluate((c) => {
      const x = c.getContext("2d"),
        d = x.getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 0) n++;
      return n;
    });
    ok(pixels > 500, "actualchartpixels");
    await page.locator("#slice-details summary").click();
    ok((await page.locator("#slice tr").count()) === 61, "61slice");
    await page.locator("#coefficients summary").click();
    ok((await page.locator("#weights tr").count()) === 35, "35weights");
    ok(
      (await page.locator("#weights td").count()) === 385,
      "350coefficients+35powers",
    );
    const weights = await page
      .locator("#weights tr")
      .first()
      .locator("td")
      .allTextContents();
    ok(
      weights
        .slice(1)
        .every((v, i) => Math.abs(Number(v) - expected.weights[i][0]) < 1e-8),
      "realweights",
    );
    const allWeights = await page
      .locator("#weights tr")
      .evaluateAll((rows) =>
        rows.map((row) =>
          [...row.querySelectorAll("td")]
            .slice(1)
            .map((td) => Number(td.textContent)),
        ),
      );
    ok(
      allWeights.every((row, i) =>
        row.every((v, j) => Math.abs(v - expected.weights[j][i]) < 1e-8),
      ),
      "all350realweights",
    );

    ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "wideweightstablecontained",
    );
    await page.locator("#coefficients summary").click();
    for (const greek of ["1", "2", "3", "4", "0"]) {
      await page.locator("#greek").selectOption(greek);
      ok(
        (await page.locator("#chart-caption").innerText()).includes(
          ["delta", "gamma", "vega", "theta", "rho"][Number(greek)],
        ),
        "chartchoice",
      );
    }
    await page.locator("#kind").selectOption("put");
    ok(await page.locator("#greek").isDisabled(), "dirtydisabled");
    ok(
      (await page.locator("#notice").innerText()).includes(
        "Previous point is stale",
      ),
      "dirtywarn",
    );
    await page.locator("#compare").click();
    ok(
      (await page.locator("#status").innerText()).startsWith("PUT"),
      "putapplied",
    );
    ok(
      Number(
        await page.locator("#greeks tr td:nth-child(2)").first().innerText(),
      ) < 0,
      "putdelta",
    );
    for (const [s, t, v] of [
      [85, 0.25, 15],
      [115, 2, 45],
      [97.5, 0.6875, 22.5],
    ]) {
      await page.locator("#spot").fill(String(s));
      await page.locator("#years").fill(String(t));
      await page.locator("#vol").fill(String(v));
      await page.locator("#compare").click();
      ok(!(await page.locator("#greek").isDisabled()), "validbounds");
      ok(
        (await page.locator("#notice").innerText()).startsWith("Applied."),
        "validrun",
      );
    }
    for (const [id, v] of [
      ["#spot", "84.999"],
      ["#spot", "115.001"],
      ["#spot", ""],
      ["#years", "0"],
      ["#years", "2.001"],
      ["#vol", "14.999"],
      ["#vol", "45.001"],
    ]) {
      await page.locator(id).fill(v);
      await page.locator("#compare").click();
      ok(
        (await page.locator("#notice").innerText()).includes("Not applied:"),
        "invalidrejected",
      );
      ok(await page.locator("#greek").isDisabled(), "invalidstaledisabled");
      await page.locator("#spot").fill("100");
      await page.locator("#years").fill("1");
      await page.locator("#vol").fill("30");
    }
    await page.locator("#compare").click();
    await page.locator("#greek").selectOption("1");
    const expectedSlice = await page.evaluate(async () => {
      const { fit, predict, analytic } = await import("./core.js"),
        m = fit();
      return Array.from({ length: 61 }, (_, i) => {
        const p = [85 + i * 0.5, 1, 0.3],
          a = analytic(p, "put").greeks[1],
          b = predict(m, p, "put")[1];
        return [p[0], a, b, b - a];
      });
    });
    const shownSlice = await page
      .locator("#slice tr")
      .evaluateAll((rows) =>
        rows.map((row) =>
          [...row.querySelectorAll("td")].map((td) => Number(td.textContent)),
        ),
      );
    ok(
      shownSlice.every((row, i) =>
        row.every((v, j) => Math.abs(v - expectedSlice[i][j]) < 0.00000051),
      ),
      "all61slicevaluesactual",
    );
    await page.locator("#all-holdout summary").click();
    ok((await page.locator("#holdout tr").count()) === 216, "allputholdout");
    await page.screenshot({
      path: `/tmp/tangentlab-shots/lab-${width}.png`,
      fullPage: true,
    });
    ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "finalnooverflow",
    );
    ok(errors.length === 0, errors.join(","));
    ok(failed.length === 0, failed.join(","));
    ok(third.length === 0, third.join(","));
    ok(
      (await page.evaluate(
        () => localStorage.length + sessionStorage.length,
      )) === 0,
      "nostorage",
    );
    ok((await page.context().cookies()).length === 0, "nocookies");
    await page.close();
  }
  console.log(`${count} browser checks across 5 widths PASS`);
} finally {
  await browser.close();
  if (server) server.kill();
}
