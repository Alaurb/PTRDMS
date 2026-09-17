// Node-only smoke test for exported report behavior; no browser dependency.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context2d = new Proxy({}, {get: (_, key) => key === 'createRadialGradient'
  ? () => ({addColorStop(){}}) : () => {}});
const nodes = new Map();
const node = () => ({style:{}, dataset:{}, classList:{toggle(){}},
  addEventListener(){}, appendChild(){}, after(){},
  getBoundingClientRect(){return {width:900,height:500}},
  clientWidth:900, clientHeight:500, getContext(){return context2d}});
const sandbox = {document:{querySelector(key){if(!nodes.has(key))nodes.set(key,node());return nodes.get(key)},
  querySelectorAll(){return []},createElement:node},window:{addEventListener(){}},devicePixelRatio:1};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const check = `
const initialPoints = new Set(hit.map(p=>JSON.stringify([p.plant.plant_id,p.x,p.y])));
enabled[Object.keys(colors)[0]]=false; draw();
if(!hit.every(p=>initialPoints.has(JSON.stringify([p.plant.plant_id,p.x,p.y]))))throw Error('Filtering moved observations');
coordinateMode=true;
const p=plants.find(p=>p.cubes.length),c=p.cubes[0],q=fruitPosition(p,c,0);
if(q.x!==c.x||q.y!==c.y||q.z!==c.z)throw Error('Coordinate view altered recorded coordinates');
draw();
`;
vm.runInContext(check,sandbox);
assert(nodes.get('h1').textContent.includes('Four-stage') || nodes.get('h1').textContent.includes('Green Gem'));
console.log('Report runtime OK: stable filtering, preserved coordinates, metadata title');
