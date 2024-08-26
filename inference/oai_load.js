import { check } from 'k6';
import http from 'k6/http';
import { scenario } from 'k6/execution';
import { SharedArray } from 'k6/data';

// Define configurations
const vu = __ENV.VU || 1;
const data = __ENV.DATA || 'samples.json';

const samples = new SharedArray('ShareGPT samples', function () {
  return JSON.parse(open(data));
});

export const options = {
  thresholds: {
    http_req_failed: ['rate==0'],
  },
  scenarios: {
    throughput: {
      executor: 'shared-iterations',
      vus: vu,
      iterations: samples.length,
      maxDuration: '90m',
    },
  },
};

export default function () {
  // Load ShareGPT random example
  const sample = samples[scenario.iterationInTest];
  // Create Body 
  const payload = {
    inputs: sample.inputs,
    parameters: {
      max_new_tokens: 512,
      details: true,
      sample: true,
      top_p: 0.9,
      top_k: 50,
      temperature: 0.2,
    },
  };

  const headers = { 'Content-Type': 'application/json' };
  const res = http.post("http://localhost:8080/generate", JSON.stringify(payload), {
    headers, timeout: '20m'
  });
  check(res, {
    'Post status is 200': (r) => res.status === 200,
  });
}