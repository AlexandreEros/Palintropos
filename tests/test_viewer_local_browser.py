"""Exercise Viewer 2 through the actual localhost UI in network-isolated Chromium."""
import json
import threading
import time

import pytest

from tools.browser_cdp import OfflinePage, find_browser
from tropoi.representation.visual.viewer_server import ViewerServer, ViewerSession
from test_viewer_manifest import fixture, BEN, BVE, SWE, CORE


def until(page, expression):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if page.eval(expression):
            return
        time.sleep(.12)
    pytest.fail(f"{expression}: {page.eval('document.body.innerText')}; {page.exceptions}")


def set_control(page, key, value, event="change"):
    selector = json.dumps(f'[data-id="{key}"]')
    page.eval(f"(() => {{const el=document.querySelector({selector}); el.value={json.dumps(str(value))};"
              f"el.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}));}})()")


def test_ben_to_shallow_water_and_bve_in_same_application():
    browser = find_browser()
    if not browser:
        pytest.skip("Needs a Chromium browser")
    paths = [fixture(n) for n in (BEN, SWE, BVE, CORE)]
    with ViewerServer(ViewerSession(paths)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with OfflinePage(browser, loopback_port=server.server_port) as page:
                page.call("Page.navigate", url=server.url)
                ready = 'document.querySelector(\'[data-id="png"]\')?.disabled === false'
                until(page, ready)
                assert page.eval('document.querySelector(\'[data-id="field"]\').value') == "temperature"
                set_control(page, "time", 29, "input")
                until(page, ready + ' && document.querySelector(\'[data-id="time-label"]\').textContent.includes("30")')
                # Capture the same Blob the normal Save PNG button hands to the browser.
                page.eval("""const makeURL = URL.createObjectURL.bind(URL);
                    URL.createObjectURL = blob => {window.exportBlob=blob; return makeURL(blob)};
                    HTMLAnchorElement.prototype.click = async function() {
                    const blob = window.exportBlob;
                    const pixels = await createImageBitmap(blob);
                    window.exported = [blob.type, pixels.width, pixels.height, this.download];
                }; document.querySelector('[data-id="png"]').click()""")
                until(page, "Boolean(window.exported)")
                kind, width, height, filename = page.eval("window.exported")
                assert kind == "image/png" and width >= 1000 and height >= 500
                assert "temperature" in filename and "2592000" in filename
                page.eval('document.getElementById("compare").click()')
                # Two globes (the Explorer's and an independent comparison); no flat map under either.
                until(page, 'document.querySelectorAll(\'[data-id="surface"] canvas\').length === 2')
                # Each globe shows its own live colour scale: label and the drawn range's endpoints.
                until(page, "viewer.panels.every(p => p.view.scene)")
                legends = page.eval("""viewer.panels.map(p => { const v = p.view;
                    return [v.legend.hidden, v.legendLabel.textContent, v.legendMin.textContent,
                            v.legendMax.textContent, v.scene.scale.range]; })""")
                for hidden, label, low, high, extent in legends:
                    assert not hidden and label and low and high and extent is not None
                for number, field in [(1, "phi"), (2, "zeta")]:
                    page.eval(f'''(() => {{const el=document.getElementById("runs");
                     el.selectedIndex={number}; el.dispatchEvent(new Event("change"));}})()''')
                    until(page, f'location.pathname.endsWith("/{number}/") && ' + ready)
                    page.eval(f"""(() => {{const el = document.querySelector('[data-e="quantity"]'); el.value = "{field}";
                        el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")
                    until(page, ready + f' && viewer.panels[0].key === "{field}"')
                    assert page.eval('document.querySelector(\'[data-id="vertical"]\').hidden')
                    if field == "phi":
                        assert page.eval("""(async () => {
                            const {loadFrame} = await import('./data.js');
                            const {fieldScale,surfaceLayers} = await import('./mesh.js');
                            const get = quantity => loadFrame({quantity,longitude:{role:'x'},latitude:{role:'y'},vertical:{role:'none'},time:{role:'fixed',index:3}});
                            const [phi,speed] = await Promise.all([get('phi'),get('wind_speed')]);
                            const options = {shell:true, radius:6371220, exaggeration:75};
                            const mesh = (f,key) => surfaceLayers(f,options,fieldScale(f,{...options,field:key}));
                            const [ground,fluid] = mesh(phi,'phi'), [,recoloured] = mesh(speed,'wind_speed');
                            const length = (p,k) => Math.hypot(...p.subarray(3*k,3*k+3));
                            for(let k=0;k<phi.height.length;k++) {
                                const expected = 75*(phi.height[k]-phi.terrain[k])/options.radius;
                                if(Math.abs(length(fluid.positions,k)-length(ground.positions,k)-expected)>2e-7)
                                    return false;
                            }
                            return fluid.positions.every((v,k) => v===recoloured.positions[k]) &&
                                   fluid.colors.some((v,k) => v!==recoloured.colors[k]);
                        })()""")
                page.eval('document.getElementById("runs").selectedIndex=3;document.getElementById("runs").dispatchEvent(new Event("change"))')
                # The carried configuration (a profile) cannot be answered by a recorded series: it is kept
                # and explained, and adapting it is an explicit step.
                until(page, 'location.pathname.endsWith("/3/") && '
                      '!document.querySelector(\'[data-e="incompatible"]\')?.hidden')
                page.eval('document.querySelector(\'[data-e="adapt"]\').click()')
                until(page, 'window.viewer?.explorer?.data?.description === "Recorded series" && '
                      'document.querySelector(\'[data-e="png"]\')?.disabled === false')
                assert not page.exceptions, page.exceptions
                assert all(url.startswith("http://"+server.authority+"/") or url.startswith(("data:", "blob:"))
                           for url in page.requests), page.requests
        finally:
            server.shutdown()
            thread.join()
