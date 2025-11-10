import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai'; //to be local later
const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');
const taskSelect = document.getElementById('task');

const modelFileName = './weights.bin';

function displayPartialResults(partialResults, complete) {
  output.textContent += partialResults;

  if (complete) {
    if (!output.textContent) {
      output.textContent = 'Result is empty.';
    }
    submit.disabled = false;
    submit.textContent = 'Process';
  }
}

function getTaskInstruction(task) {
  switch (task) {
    case 'Summarize':
      return `EDIT ONLY: Summarize the following text in 3-5 sentences. Do NOT add, infer, remove, or change facts. Preserve only facts present in the input. Output only the summary (no commentary):`;
    case 'Proofread':
      return `EDIT ONLY: Correct grammar, spelling, punctuation, and clarity for the following text. Do NOT add or remove factual content, consequences, or actions. Preserve original meaning and sentence count. Output only the corrected text (no commentary):`;
    case 'Paraphrase':
      return `EDIT ONLY: Paraphrase the following text conservatively. Keep meaning, tone, and approximate length identical. Do NOT add, remove, or infer any facts. Output only the paraphrase (no commentary):`;
    case 'Rewrite':
    default:
      return `EDIT ONLY: Rewrite the following text for clarity and flow WITHOUT adding, removing, or inferring any facts, consequences, or actions. Preserve exact meaning and sentence count. Slightly rephrase wording and adjust punctuation for improved flow and clarity, even if grammar is already correct. Output only the rewritten text (no commentary):`;
  }
}

function buildStrictPrompt(task, text) {
  switch (task) {
    case 'Summarize':
      return `EDIT ONLY: Summarize the following text in 3-5 sentences. Do NOT add, infer, remove, or change facts. Preserve only facts present in the input. Output only the summary (no commentary): ${text}`;
    case 'Proofread':
      return `EDIT ONLY: Correct grammar, spelling, punctuation, and clarity for the following text. Do NOT add or remove factual content, consequences, or actions. Preserve original meaning and sentence count. Output only the corrected text (no commentary): ${text}`;
    case 'Paraphrase':
      return `EDIT ONLY: Paraphrase the following text conservatively. Keep meaning, tone, and approximate length identical. Do NOT add, remove, or infer any facts. Output only the paraphrase (no commentary): ${text}`;
    case 'Rewrite':
    default:
      return `EDIT ONLY: Rewrite the following text for clarity and flow WITHOUT adding, removing, or inferring any facts, consequences, or actions. Preserve exact meaning and sentence count. Change only wording and punctuation as needed for grammar and clarity. Output only the rewritten text (no commentary): ${text}`;
  }
}


async function runDemo() {
  submit.disabled = true;
  submit.textContent = 'Loading model...';

  const genaiFileset = await FilesetResolver.forGenAiTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm'
  );

  let llmInference;
  try {
    llmInference = await LlmInference.createFromOptions(genaiFileset, {
      baseOptions: { modelAssetPath: modelFileName },
      temperature: 0.9, // deterministic if <=0.5
      topK: 20,         // restrict long tail creativity
      topP: 0.8,
      maxTokens: 600,
      stopSequences: ['\n\nINPUT:', '\n\nOUTPUT:', '\n\nEDIT ONLY:'],
    });

    submit.disabled = false;
    submit.textContent = 'Process';
  } catch (error) {
    alert('Failed to initialize the AI model. Ensure "weights.bin" is available and served correctly.');
    console.error(error);
    submit.textContent = 'Init failed';
    return;
  }

  const rewriteInstruction = getTaskInstruction('Rewrite');

  const fewShot = [
    {
      input: 'The system failed to start due to a memory allocation error. Diagnostics showed the garbage collector could not free inactive references.',
      output: 'The system failed to start because of a memory allocation error. Diagnostics showed the garbage collector could not free inactive references.'
    },
    {
      input: 'Calibration completed; sensors returned stable readings within tolerance.',
      output: 'Calibration completed, and the sensors returned stable readings within tolerance.'
    }
  ].map(pair => `${rewriteInstruction}\nINPUT: ${pair.input}\nOUTPUT: ${pair.output}`).join('\n\n');

  submit.onclick = () => {
    if (!input.value.trim()) {
      alert('Please enter some text to process.');
      return;
    }

    output.textContent = '';
    submit.disabled = true;
    submit.textContent = 'Generating...';

    const task = taskSelect.value;
    const text = input.value;

    const strictPrompt = buildStrictPrompt(task, text);

    const taskInstruction = getTaskInstruction(task);

    const fullPrompt = `${fewShot}\n\n${taskInstruction}\nINPUT: ${text}\nOUTPUT:`;

    try {
      llmInference.generateResponse(fullPrompt, displayPartialResults);
    } catch (err) {
      try {
        llmInference.generateResponse(strictPrompt, displayPartialResults);
      } catch (err2) {
        console.error('generateResponse failed with both prompt forms', err, err2);
        alert('Model generation failed. Check console for details.');
        submit.disabled = false;
        submit.textContent = 'Process';
      }
    }
  };
}

runDemo();