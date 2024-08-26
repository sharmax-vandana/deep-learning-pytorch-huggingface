import { check } from 'k6';
import http from 'k6/http';
import { scenario } from 'k6/execution';

// curl -L -o samples.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
// 1. run TGI then k6s
// k6 run -e VU=1 -e DATA=samples.json oai_load.js

// Define configurations
const vu = __ENV.VU || 1;
const data = __ENV.DATA || 'samples.json';
const host = __ENV.HOST || '127.0.0.1:8080';
const model_id = __ENV.MODEL_ID || 'tgi';
const max_new_tokens = __ENV.MAX_NEW_TOKENS || 200;

const samples = JSON.parse(open(data))

export function get_options() {
  return {
    scenarios: {
      load_test: {
        executor: 'constant-arrival-rate',
        duration: '30s',
        preAllocatedVUs: vu,
        rate: 1,
        timeUnit: '1s',
      },
    },
  };
}

// export function handleSummary(data) {
//   return {
//     'summary.json': JSON.stringify(data),
//   };
// }


function generate_payload(data, max_new_tokens) {
  let input = data["conversations"][0]["value"];
  if (input.length > 500) {
    input = input.slice(0, 500)
  }
  return {
    "messages": [{ "role": "user", "content": input }],
    "temperature": 0,
    "model": `${model_id}`,
    "max_tokens": max_new_tokens,
    "stream": true,
  };
}

export const options = get_options();


export default function run() {
  const headers = { 'Content-Type': 'application/json' };
  const query = samples[scenario.iterationInTest % samples.length];
  const payload = JSON.stringify(generate_payload(query, max_new_tokens));

  const url = `http://${host}/v1/chat/completions`;

  const startTime = Date.now();
  // let firstTokenTime = null;
  // let lastTokenTime = null;
  // let tokensCount = 0;
  // let response = ""

  // TODO: ADD SSE 
  const res = http.post(url, payload, { headers: headers });

  if (res.status >= 400 && res.status < 500) {
    return;
  }

  check(res, {
    'Post status is 200': (res) => res.status === 200,
  });

}


// const res = sse.open(url, params, function (client) {
//   client.on('event', function (event) {
//     // console.log(event.data)
//     if (parseInt(event.id) === 4) {
//       client.close()
//     }
//     if (event.data.includes("[DONE]")) {
//       return
//     }
//     try {
//       const data = JSON.parse(event.data);
//       const content = data['choices'][0]['delta']['content']
//       if (content !== undefined) {
//         response += data['choices'][0]['delta']['content']
//       }
//       tokensCount += 1;

//       // Measure time to first token
//       if (!firstTokenTime) {
//         firstTokenTime = Date.now();
//         timeToFirstToken.add(firstTokenTime - startTime);
//       }

//       // Measure inter-token latency
//       const currentTime = Date.now();
//       if (lastTokenTime) {
//         interTokenLatency.add(currentTime - lastTokenTime);
//       }
//       lastTokenTime = currentTime;

//       if ('finish_reason' in data['choices'][0]) {
//         const endTime = Date.now();
//         const deltaMs = endTime - startTime;
//         endToEndLatency.add(deltaMs)
//         requestThroughput.add(1);
//         tokenThroughputPerSec.add(tokensCount / deltaMs * 1000);
//         tokensReceived.add(tokensCount);
//         // console.log(response)
//       }
//     } catch (e) {
//       console.error('An unexpected error occurred: ', e)
//       console.log(event)
//     }
//   })

//   client.on('error', function (e) {
//     console.log('An unexpected error occurred: ', e.error())
//   })
// })
