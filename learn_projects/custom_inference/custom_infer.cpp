#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    // 1. Model Configuration
    std::string model_path = "../llama.cpp/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
    std::string prompt = "Explain virtual memory in 1 sentence.";
    int ngl = 99; // Offload all layers to Metal GPU
    int n_predict = 64; // Max tokens to generate

    printf("========================================================\n");
    printf("🚀 CUSTOM C++ LLM INFERENCE ENGINE (llama.h API)\n");
    printf("========================================================\n\n");

    // 2. Load Metal GPU Backends
    ggml_backend_load_all();

    // 3. Step 1: Load Model via mmap
    printf("📦 Step 1: Loading GGUF Model via mmap()...\n");
    auto t_start_load = std::chrono::high_resolution_clock::now();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "❌ Error: Could not load model from %s\n", model_path.c_str());
        return 1;
    }
    
    auto t_end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> load_duration = t_end_load - t_start_load;
    printf("   ✅ Model mapped in %.2f ms!\n", load_duration.count());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("   📊 Vocabulary Size: %d tokens\n\n", n_vocab);

    // 4. Step 2: Tokenization (Text -> Integer Token IDs)
    printf("🔤 Step 2: Tokenizing Prompt: \"%s\"\n", prompt.c_str());
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

    printf("   🔢 Token IDs generated: [ ");
    for (auto id : prompt_tokens) {
        printf("%d ", id);
    }
    printf("] (%zu tokens)\n\n", prompt_tokens.size());

    // 5. Step 3: Initialize Context & Allocate KV-Cache
    printf("🧠 Step 3: Allocating Context & KV-Cache in RAM...\n");
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "❌ Error: Failed to create context\n");
        return 1;
    }
    printf("   ✅ KV-Cache allocated for context size: %d\n\n", ctx_params.n_ctx);

    // 6. Step 4: Initialize Greedy Sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // 7. Step 5: The Prefill Phase (Prompt Evaluation)
    printf("⚡ Step 5: Running Prefill (Evaluating %zu tokens in parallel)...\n", prompt_tokens.size());
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "❌ Error: Prefill decode failed\n");
        return 1;
    }
    
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prefill_duration = t_prefill_end - t_prefill_start;
    printf("   ✅ Prefill completed in %.2f ms (TTFT)!\n\n", prefill_duration.count());

    // 8. Step 6: The Autoregressive Generation Loop
    printf("💬 Generated Output: \n\n");
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int n_decoded = 0;

    for (int i = 0; i < n_predict; ++i) {
        // Sample next token ID
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for End-of-Generation (EOG)
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Convert Token ID -> Character Piece
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
        }

        // Prepare next batch with the 1 newly generated token
        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "❌ Decode failed\n");
            break;
        }
        n_decoded++;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decode_seconds = t_decode_end - t_decode_start;

    printf("\n\n--------------------------------------------------------\n");
    printf("📊 Inference Performance Stats:\n");
    printf("   - Tokens Generated: %d\n", n_decoded);
    printf("   - Generation Time:  %.3f seconds\n", decode_seconds.count());
    printf("   - Generation Speed: %.2f tokens/second\n", n_decoded / decode_seconds.count());
    printf("========================================================\n");

    // 9. Clean up memory
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
