const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../runtime/minimax_h3/console.html'), 'utf8');
const source = html.match(/<script>([\s\S]*)<\/script>/)[1];

function mount(initial) {
  const nodes = new Map(), timers = new Map(), requests = [];
  let timerId = 0, now = 1000000;
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {id, dataset: {}, value: '', textContent: '', disabled: false,
      listeners: {}, classList: {toggle() {}}, addEventListener(name, callback) {this.listeners[name] = callback;},
      setAttribute() {}, removeAttribute() {}, append() {}, replaceChildren() {}, prepend() {},
      scrollHeight: 1, scrollTop: 0, clientHeight: 1});
    return nodes.get(id);
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) node(match[1]);
  const actions = [...html.matchAll(/data-action="([^"]+)"/g)].map(match => {
    const button = node('action-' + match[1]); button.dataset.action = match[1]; return button;
  });
  node('log-source').value = 'diagnostics'; node('log-lines').value = '100';
  node('follow-logs').checked = true;
  const complete = {ui: {connected: true, bootstrap_running: true, busy: true,
    startup_delivery: {active: false, status: 'api_ready'}, operations: [{action: 'bootstrap'}]},
    state: {api_ready: true, model_ready: false, observation_stale: false}, logs: {source: 'diagnostics', text: 'API delivered'}};
  class ClockDate extends Date {static now() {return now;}}
  const window = {H3_TRANSPORT: async (action, data) => {requests.push({action, data}); return complete;},
    addEventListener() {}, crypto: {randomUUID: () => 'test-request'}};
  const context = vm.createContext({window, document: {getElementById: node, querySelectorAll: () => actions,
    createElement: () => node('generated-' + nodes.size), addEventListener() {}, hidden: false},
    navigator: {}, URL, Blob, Date: ClockDate, Intl, Uint8Array, console,
    setTimeout: (callback, delay) => {const id = ++timerId; timers.set(id, {callback, delay}); return id;},
    clearTimeout: id => timers.delete(id)});
  vm.runInContext(source.replace('__CALLBACK_NAME__', '"test"').replace('__INITIAL_STATE__', JSON.stringify(initial)), context);
  return {node, actions, timers, requests, advance(value) {now = value;}};
}

test('foreground delivery suppresses queued RPC timeouts and disables RPC controls', async () => {
  const ui = mount({ui: {connected: true, startup_delivery: {active: true, deadline_at: 1180}}});
  assert.match(ui.node('next-step').textContent, /停止本单元/);
  assert.match(ui.node('connection-state').textContent, /宿主推送/);
  assert.ok(ui.actions.every(button => button.disabled));
  assert.equal(ui.node('refresh-board').disabled, true);
  await ui.node('refresh-board').listeners.click();
  assert.equal(ui.requests.length, 0);
  const scheduled = [...ui.timers.values()];
  assert.equal(scheduled.length, 1);
  assert.ok(scheduled[0].delay >= 180000);
  ui.advance(1186000);
  scheduled[0].callback();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.node('action-stop').disabled, false);
  assert.equal(ui.node('refresh-board').disabled, false);
  assert.doesNotMatch(ui.node('connection-state').textContent, /超时|待确认/);
});

test('completed handoff restores normal short polling while model still prepares', () => {
  const ui = mount({ui: {connected: true, startup_delivery: {active: false, status: 'api_ready'},
    bootstrap_running: true, busy: true}, state: {api_ready: true, model_ready: false}});
  assert.equal([...ui.timers.values()][0].delay, 2000);
  assert.equal(ui.node('action-stop').disabled, false);
  assert.equal(ui.node('action-bootstrap').disabled, true);
});

const conditionFixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures/live-condition-projection.json'),'utf8'));
for(const fixture of conditionFixtures){
  test('shared condition projection: '+fixture.name,()=>{
    const ui=mount({ui:{connected:true},state:fixture.state});
    assert.equal(ui.node('headline').textContent,fixture.headline);
    if(fixture.detail)assert.ok(ui.node('next-step').textContent.includes(fixture.detail));
    assert.equal(ui.node('notice-zone').textContent,'');
  });
}
