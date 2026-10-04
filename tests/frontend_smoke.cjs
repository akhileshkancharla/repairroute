const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'frontend', 'app.js'), 'utf8');
const context = {document: {addEventListener() {}}};
vm.runInNewContext(`${source}\nthis.parseCsvForTest = parseCsv;`, context);

const example = fs.readFileSync(path.join(root, 'artifacts', 'sample_batch.csv'), 'utf8');
const rows = context.parseCsvForTest(example);
assert.equal(rows.length, 5);
assert.equal(Object.keys(rows[0]).length, 170);
assert.ok(Object.values(rows[0]).some(value => value === null));
assert.throws(() => context.parseCsvForTest('a,b\n1,2'), /170 distinct/);
console.log('Frontend CSV parsing checks passed.');
