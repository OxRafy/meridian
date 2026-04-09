import { createMultiLLM } from './tools/llm-adapter/index.js';

console.log('🧪 Testing Multi-Provider LLM Adapter...\n');

import('dotenv').then(({ default: dotenv }) => {
  dotenv.config();
  
  const config = {
    providers: [
      {
        type: 'openrouter',
        name: 'primary',
        model: 'minimax/minimax-m2.7',
        apiKey: process.env.OPENROUTER_API_KEY || 'your_key_here',
        timeout: 5 * 60 * 1000,
      },
      {
        type: 'ollama',
        name: 'backup',
        model: 'minimax/minimax-m2.7',
        baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1',
        apiKey: 'ollama',
      },
    ],
    defaultModel: 'minimax/minimax-m2.7',
    fallbackModel: 'stepfun/step-3.5-flash:free',
    maxRetries: 3,
  };

  console.log('📦 Config loaded:', Object.keys(config).join(', '));

  const multiLLM = createMultiLLM(config);
  
  console.log('\n✅ MultiLLMManager initialized!');
  
  // Inspect structure
  console.log('\n🔍 Inspecting MultiLLMManager structure...');
  console.log('   - Type:', typeof multiLLM);
  console.log('   - Methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(multiLLM)).filter(m => !m.startsWith('_')).join(', '));
  
  // Try to get default provider
  console.log('\n🔄 Testing inference...\n');
  
  const prompt = "Hello! Can you introduce yourself? Keep it short and friendly.";
  console.log('📝 Prompt:', prompt);
  console.log('\n⏳ Sending request...\n');

  return multiLLM.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: prompt },
    ],
    model: config.defaultModel,
  })
    .then(result => {
      console.log('✅ Response received!');
      console.log('📝 Response:', result.choices[0].message.content);
      console.log('\n🎉 Multi-LLM Adapter is working!');
      console.log('\n📋 Available configurations:');
      console.log('   - Primary:', config.defaultModel);
      console.log('   - Fallback:', config.fallbackModel);
      console.log('   - Max Retries:', config.maxRetries);
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Error:', error.message);
      console.error('💡 This is expected if no API key is configured.');
      console.log('\n📋 To test properly:');
      console.log('   1. Set OPENROUTER_API_KEY in .env');
      console.log('   2. OR set OLLAMA_BASE_URL and OLLAMA_MODEL');
      console.log('   3. Then run: node test-multi-llm.js');
      process.exit(1);
    });
});
