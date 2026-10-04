const $ = (id) => document.getElementById(id);
const fmt = (value) => new Intl.NumberFormat('en-US').format(Math.round(value));
const percent = (value) => `${(100 * value).toFixed(1)}%`;
const state = { metadata: null, curve: null, capacity: 597, queueExpanded: false, queueOffset: 0, scoreRows: [], selectedFile: null, requestId: 0 };

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body ?? 'Request failed');
    throw new Error(detail);
  }
  return body;
}

function error(message) {
  const banner = $('global-error');
  banner.textContent = message;
  banner.classList.remove('hidden');
}
function clearError() { $('global-error').classList.add('hidden'); }

function setView(view) {
  if (!['planner', 'batch', 'evidence'].includes(view)) view = 'planner';
  document.querySelectorAll('.view').forEach((item) => item.classList.toggle('active', item.id === view));
  document.querySelectorAll('.nav-link').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
  $('topbar-title').textContent = {planner:'Inspection planner', batch:'Score a batch', evidence:'Model evidence'}[view];
  $('sidebar').classList.remove('open');
  $('menu-toggle').setAttribute('aria-expanded', 'false');
  document.title = `RepairRoute · ${$('topbar-title').textContent}`;
  history.replaceState(null, '', `#${view}`);
  window.scrollTo({top:0, behavior:'instant'});
}

function tableRows(container, rows) {
  container.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const value of row) {
      const td = document.createElement('td');
      td.textContent = String(value);
      tr.append(td);
    }
    container.append(tr);
  }
}

function setCapacity(value) {
  const slider = $('capacity-range');
  state.capacity = Math.max(0, Math.min(Number(slider.max), Number(value) || 0));
  if (state.queueOffset >= state.capacity) state.queueOffset = 0;
  slider.value = String(state.capacity);
  $('capacity-output').textContent = fmt(state.capacity);
  if (state.metadata) $('capacity-percent').textContent = `${(100 * state.capacity / state.metadata.historical_records).toFixed(1)}% of the historical batch`;
  slider.style.background = `linear-gradient(to right, #087E83 ${100 * state.capacity / Number(slider.max)}%, #D6E0E2 0)`;
  updatePlanner();
}

async function updatePlanner() {
  if (!state.metadata) return;
  const requestId = ++state.requestId;
  try {
    const capacity = state.capacity;
    const [decision, queue] = await Promise.all([
      api(`/v1/historical/decision?capacity=${capacity}`),
      api(`/v1/historical/queue?capacity=${capacity}&limit=${state.queueExpanded ? 100 : 5}&offset=${state.queueExpanded ? state.queueOffset : 0}`),
    ]);
    if (requestId !== state.requestId) return;
    clearError();
    renderPlanner(decision, queue);
  } catch (exc) { if (requestId === state.requestId) error(`Planner could not update: ${exc.message}`); }
}

function renderPlanner(decision, queue) {
  const m = state.metadata, o = decision.outcome, k = decision.capacity;
  $('kpi-inspections').textContent = fmt(o.inspections);
  $('kpi-caught').textContent = fmt(o.caught);
  $('kpi-missed').textContent = fmt(o.missed);
  $('kpi-unnecessary').textContent = fmt(o.unnecessary);
  $('kpi-recall').textContent = `${percent(o.recall)} of ${fmt(m.historical_aps_faults)}`;
  $('chart-cost').textContent = fmt(o.cost);
  $('chart-capacity').textContent = `at ${fmt(k)} inspections`;
  $('next-record').textContent = decision.next_record_id ? `Next in line: ${decision.next_record_id}` : 'Every record is selected';
  tableRows($('queue-rows'), queue.records.length ? queue.records.map((r) => [String(r.rank).padStart(2, '0'), r.record_id, r.priority_score.toFixed(3)]) : [['—','No records selected','—']]);
  $('queue-expand').textContent = state.queueExpanded ? 'Show first five ←' : `View all ${fmt(k)} ranked records →`;
  $('queue-expand').disabled = k === 0;
  $('queue-pagination').classList.toggle('hidden', !state.queueExpanded || k <= 100);
  $('queue-page-label').textContent = `${fmt(state.queueOffset + 1)}–${fmt(Math.min(state.queueOffset + 100, k))} of ${fmt(k)}`;
  $('queue-prev').disabled = state.queueOffset === 0;
  $('queue-next').disabled = state.queueOffset + 100 >= k;
  $('queue-download').disabled = k === 0;

  const pos = m.historical_aps_faults, neg = m.historical_records - pos;
  const randomCaught = pos * k / m.historical_records;
  const randomMissed = pos - randomCaught;
  const randomUnnecessary = neg * k / m.historical_records;
  tableRows($('comparison-rows'), [
    [`Ranked at ${fmt(k)}`, fmt(k), fmt(o.caught), fmt(o.missed), fmt(o.unnecessary), fmt(o.cost)],
    ['Inspect all', fmt(m.historical_records), fmt(pos), '0', fmt(neg), fmt(decision.comparisons.inspect_all.cost)],
    ['Inspect none', '0', '0', fmt(pos), '0', fmt(decision.comparisons.inspect_none.cost)],
    [`Random at ${fmt(k)} · expected`, fmt(k), `≈${fmt(randomCaught)}`, `≈${fmt(randomMissed)}`, `≈${fmt(randomUnnecessary)}`, `≈${fmt(decision.comparisons.random_same_capacity_expected_cost)}`],
  ]);
  renderChart(state.curve?.points ?? [], k, o.cost);
}

function svgNode(tag, attributes = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}
function renderChart(points, selectedCapacity, selectedCost) {
  if (!points.length) return;
  const svg = $('cost-chart'); svg.replaceChildren();
  const W=700,H=280,L=42,R=14,T=16,B=37, PW=W-L-R,PH=H-T-B;
  const maxX=state.curve.max_capacity, costs=points.map((p)=>p.cost), minY=Math.min(...costs), maxY=Math.max(...costs);
  const pad=Math.max(5000,(maxY-minY)*.08), low=Math.max(0,minY-pad), high=maxY+pad;
  const x=(v)=>L+PW*v/maxX, y=(v)=>T+PH*(high-v)/(high-low);
  for(let i=0;i<4;i++){
    const gy=T+PH*i/3;
    svg.append(svgNode('line',{x1:L,y1:gy,x2:W-R,y2:gy,stroke:'#D6E0E2','stroke-width':'1'}));
  }
  const d=points.map((p,i)=>`${i?'L':'M'}${x(p.capacity).toFixed(1)},${y(p.cost).toFixed(1)}`).join(' ');
  svg.append(svgNode('path',{d,fill:'none',stroke:'#3D6780','stroke-width':'4','stroke-linecap':'round','stroke-linejoin':'round'}));
  const sx=x(Math.min(selectedCapacity,maxX)),sy=y(selectedCost);
  svg.append(svgNode('line',{x1:sx,y1:T,x2:sx,y2:T+PH,stroke:'#087E83','stroke-dasharray':'4 5','stroke-width':'1'}));
  svg.append(svgNode('circle',{cx:sx,cy:sy,r:8,fill:'#fff',stroke:'#087E83','stroke-width':'4'}));
  for(const [value,label] of [[0,'0'],[maxX/2,fmt(maxX/2)],[maxX,`${fmt(maxX)} inspections`]]){
    const tx=svgNode('text',{x:x(value),y:H-9,fill:'#52666F','font-size':'13','text-anchor':value===0?'start':value===maxX?'end':'middle'});tx.textContent=label;svg.append(tx);
  }
  const title=svgNode('title');title.textContent=`At ${fmt(selectedCapacity)} inspections, benchmark cost is ${fmt(selectedCost)} units`;svg.append(title);
}

function renderEvidence(m) {
  const cm=m.test_confusion_matrix;
  $('evidence-recall').textContent=percent(m.test_recall);
  $('evidence-precision').textContent=percent(m.test_precision);
  $('evidence-f1').textContent=percent(m.test_f1);
  $('matrix-tp').textContent=fmt(cm.true_positive);
  $('matrix-fp').textContent=fmt(cm.false_positive);
  $('matrix-fn').textContent=fmt(cm.false_negative);
  $('matrix-tn').textContent=fmt(cm.true_negative);
  $('batch-count').textContent=fmt(m.historical_records);
  $('batch-aps').textContent=fmt(m.historical_aps_faults);
}

function parseCsv(text) {
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i];
    if(ch==='"'){
      if(quoted && text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;
    }else if(ch===','&&!quoted){row.push(field);field='';}
    else if((ch==='\n'||ch==='\r')&&!quoted){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(field);field='';if(row.some(v=>v.trim()!==''))rows.push(row);row=[];}
    else field+=ch;
  }
  if(quoted)throw new Error('CSV has an unclosed quote.');
  row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);
  if(rows.length<2)throw new Error('CSV must have a header and at least one record.');
  const headers=rows.shift().map(v=>v.trim().replace(/^\uFEFF/,''));
  if(headers.length!==170 || new Set(headers).size!==headers.length)throw new Error('Expected 170 distinct Scania feature columns. Use the example CSV for the schema.');
  return rows.map((values,index)=>{
    if(values.length!==headers.length)throw new Error(`Row ${index+2} has ${values.length} columns; expected ${headers.length}.`);
    const record={};for(let i=0;i<headers.length;i++){
      const raw=values[i].trim();const number=raw===''||raw.toLowerCase()==='na'?null:Number(raw);
      if(number!==null&&!Number.isFinite(number))throw new Error(`Row ${index+2}, ${headers[i]} is not a finite number.`);
      record[headers[i]]=number;
    }
    return record;
  });
}

function onFile(file) {
  if(!file)return;
  if(!file.name.toLowerCase().endsWith('.csv')){error('Choose a CSV file.');return;}
  clearError();state.selectedFile=file;state.scoreRows=[];
  $('file-name').textContent=file.name;
  $('file-summary').textContent=`${(file.size/1024).toFixed(1)} KB · Ready to check and score`;
  $('score-button').disabled=false;$('result-download').disabled=true;
  $('score-status').textContent='File ready. Select Score batch to run the trained model.';
  $('result-rows').replaceChildren();
}

async function scoreSelectedFile() {
  const file=state.selectedFile;if(!file)return;
  const button=$('score-button');button.disabled=true;clearError();
  try{
    $('score-status').textContent='Reading and validating CSV…';
    const records=parseCsv(await file.text());
    $('file-summary').textContent=`${fmt(records.length)} rows checked · Scoring with saved model`;
    const results=[];
    for(let start=0;start<records.length;start+=1000){
      $('score-status').textContent=`Scoring records ${fmt(start+1)}–${fmt(Math.min(start+1000,records.length))} of ${fmt(records.length)}…`;
      const batch=await api('/v1/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({records:records.slice(start,start+1000)})});
      for(const item of batch.records)results.push({index:start+item.input_index,score:item.priority_score});
    }
    results.sort((a,b)=>b.score-a.score||a.index-b.index);
    state.scoreRows=results.map((item,rank)=>({rank:rank+1,recordId:`UPLOAD-${String(item.index+1).padStart(5,'0')}`,score:item.score}));
    tableRows($('result-rows'),state.scoreRows.slice(0,20).map(r=>[String(r.rank).padStart(2,'0'),r.recordId,r.score.toFixed(3),'Inspect in order']));
    $('score-status').textContent=`Scored ${fmt(results.length)} records. Showing the first 20; download the complete ranked list.`;
    $('file-summary').textContent=`${fmt(results.length)} rows scored · Ready to download`;
    $('result-download').disabled=false;
  }catch(exc){error(`Batch could not be scored: ${exc.message}`);$('score-status').textContent='No results produced.';}
  finally{button.disabled=false;}
}

function download(filename, text) {
  const blob=new Blob([text],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function downloadQueue() {
  const rows=[];for(let offset=0;offset<state.capacity;offset+=500){const result=await api(`/v1/historical/queue?capacity=${state.capacity}&limit=500&offset=${offset}`);rows.push(...result.records);}
  download('repairroute-inspection-queue.csv','Rank,Record ID,APS priority score\n'+rows.map(r=>`${r.rank},${r.record_id},${r.priority_score}`).join('\n'));
}

async function init() {
  document.querySelectorAll('.nav-link').forEach((button)=>button.addEventListener('click',()=>setView(button.dataset.view)));
  $('menu-toggle').addEventListener('click',()=>{const open=$('sidebar').classList.toggle('open');$('menu-toggle').setAttribute('aria-expanded',String(open));});
  $('capacity-range').addEventListener('input',(event)=>setCapacity(event.target.value));
  $('capacity-down').addEventListener('click',()=>setCapacity(state.capacity-1));
  $('capacity-up').addEventListener('click',()=>setCapacity(state.capacity+1));
  $('queue-expand').addEventListener('click',()=>{state.queueExpanded=!state.queueExpanded;state.queueOffset=0;updatePlanner();});
  $('queue-prev').addEventListener('click',()=>{state.queueOffset=Math.max(0,state.queueOffset-100);updatePlanner();});
  $('queue-next').addEventListener('click',()=>{state.queueOffset+=100;updatePlanner();});
  $('queue-download').addEventListener('click',()=>downloadQueue().catch(exc=>error(`Download failed: ${exc.message}`)));
  $('batch-file').addEventListener('change',(event)=>onFile(event.target.files[0]));
  const zone=$('drop-zone');
  zone.addEventListener('dragover',(event)=>{event.preventDefault();zone.classList.add('dragover');});
  zone.addEventListener('dragleave',()=>zone.classList.remove('dragover'));
  zone.addEventListener('drop',(event)=>{event.preventDefault();zone.classList.remove('dragover');onFile(event.dataTransfer.files[0]);});
  $('score-button').addEventListener('click',scoreSelectedFile);
  $('result-download').addEventListener('click',()=>download('repairroute-scored-batch.csv','Rank,Record ID,APS priority score\n'+state.scoreRows.map(r=>`${r.rank},${r.recordId},${r.score}`).join('\n')));
  setView(location.hash.slice(1)||'planner');
  try{
    const [metadata,curve]=await Promise.all([api('/v1/metadata'),api('/v1/historical/curve?points=201&max_capacity=1600')]);
    state.metadata=metadata;state.curve=curve;state.capacity=Math.min(metadata.fixed_test_policy.inspections,1600);
    $('capacity-range').max=String(Math.min(1600,metadata.historical_records));
    renderEvidence(metadata);setCapacity(state.capacity);
  }catch(exc){error(`RepairRoute could not load its data: ${exc.message}`);}
}
document.addEventListener('DOMContentLoaded',init);
