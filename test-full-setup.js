import { createMultiLLM } from './tools/llm-adapter/index.js';
import { config } from './config.js';

console.log('🧪 Testing Full Multi-LLM Setup...\n');

import('dotenv').then(({ default: dotenv }) => {
  dotenv.config();
  
  console.log('📦 Loading config...');
  console.log('   - dryRun:', config.dryRun);
  console.log('   - defaultModel:', config.llmModel || 'minimax/minimax-m2.7');
  console.log('   - fallbackModel:', config.fallbackModel || 'stepfun/step-3.5-flash:free');
  console.log('   - maxRetries:', config.maxRetries || 3);
  console.log('   - providers:', config.providers ? config.providers.length : 2);

  const multiLLM = createMultiLLM(config);
  
  console.log('\n✅ MultiLLMManager initialized!');
  console.log('   - Methods available:', Object.getOwnPropertyNames(Object.getPrototypeOf(multiLLM)).filter(m => !m.startsWith('_')).join(', '));

  // Test 1: Health check
  console.log('\n🔄 Test 1: Health Check...');
  return multiLLM.healthCheckAll()
    .then(health => {
      console.log('   ✅ Health check passed!');
      console.log('   - Providers status:', health.map(h => `${h.name} (${h.status})`).join(', '));
      
      // Test 2: List providers
      console.log('\n🔄 Test 2: List Providers...');
      return multiLLM.listProviders();
    })
    .then(providers => {
      console.log('   ✅ Providers loaded:', providers.length);
      console.log('   - Types:', providers.map(p => p.type).join(', '));
      
      // Test 3: Chat with fallback
      console.log('\n🔄 Test 3: Chat Test (with fallback)...');
      const prompt = "Hello! Can you introduce yourself? Keep it short and friendly.";
      console.log('   - Prompt:', prompt);
      
      return multiLLM.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: prompt },
        ],
        model: config.defaultModel,
      });
    })
    .then(result => {
      console.log('   ✅ Response received!');
      console.log('   - Model used:', result.model || 'auto');
      console.log('   - Response length:', (result.choices && result.choices[0] ? result.choices[0].message.content.length : 0));
      console.log('   - Response:', result.choices?.[0]?.message?.content?.substring(0, 100) || 'Empty response');
      console.log('\n🎉 Multi-LLM Setup Complete!');
      console.log('\n📋 Summary:');
      console.log('   - Primary Provider:', config.providers?.[0]?.name || 'openrouter');
      console.log('   - Fallback Provider:', config.fallbackModel || 'stepfun/step-3.5-flash:free');
      console.log('   - Max Retries:', config.maxRetries || 3);
      console.log('   - Dry Run:', config.dryRun);
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Error:', error.message);
      console.error('💡 Check your API keys in .env file');
      console.log('\n📋 To fix:');
      console.log('   1. Set OPENROUTER_API_KEY in .env for primary provider');
      console.log('   2. OR set OLLAMA_BASE_URL and OLLAMA_MODEL for local backup');
      process.exit(1);
    });
});
