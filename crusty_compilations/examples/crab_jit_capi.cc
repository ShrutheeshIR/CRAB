#include <jit/compiler.hh>
#include <jit/object_cache.hh>
#include <jit/session.hh>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

struct CrabJitContext
{
    std::shared_ptr<crab::jit::DiskObjectCache> cache;
    crab::jit::ClangCompiler compiler;
    std::unique_ptr<crab::jit::JitSession> session;
    std::unordered_map<std::string, void *> symbols;
};

extern "C"
{
    CrabJitContext *crab_jit_create()
    {
        auto ctx = new CrabJitContext();
        ctx->cache = std::make_shared<crab::jit::DiskObjectCache>(crab::jit::default_cache_dir());
        ctx->session = std::make_unique<crab::jit::JitSession>(ctx->cache);
        return ctx;
    }

    void crab_jit_destroy(CrabJitContext *ctx)
    {
        delete ctx;
    }

    int crab_jit_compile(CrabJitContext *ctx, const char *source, const char *module_id)
    {
        if (!ctx || !source || !module_id)
        {
            // Complain with failure
            llvm::errs() << "crab::jit: invalid arguments to crab_jit_compile\n";
            return -1;
        }

        crab::jit::CompileOptions opts;
        // The disk cache is keyed by opts.module_id alone. If we used the caller's
        // module_id verbatim, editing the source (e.g. regenerating a kernel template)
        // without also changing module_id would silently hit a stale cached .o forever.
        // Fold a content hash (source + flags) into the key so any source/flag change
        // invalidates the cache, while keeping the caller's id as a readable prefix.
        const std::string cache_key =
            std::string(module_id) + "-" + crab::jit::hash_source(source, opts);
        opts.module_id = cache_key;

        auto cached_obj = ctx->cache->load_object(cache_key);
        if (cached_obj)
        {
            if (auto err = ctx->session->add_object_file(std::move(cached_obj)))
            {
                llvm::consumeError(std::move(err));
                llvm::errs() << "crab::jit: add_object_file failed: " << err << "\n";
                return -2;
            }
            return 1; // Cache hit
        }

        try
        {
            auto tsm = ctx->compiler.compile(source, opts);
            if (auto err = ctx->session->add_module(std::move(tsm)))
            {
                llvm::consumeError(std::move(err));
                llvm::errs() << "crab::jit: add_module failed\n";
                return -3;
            }
        }
        catch (const std::exception &e)
        {
            // llvm::consumeError(std::move(e));
            llvm::errs() << "crab::jit: compile failed: " << e.what() << "\n";
            return -4;
        }
        return 0; // compiled fresh
    }

    void *crab_jit_lookup(CrabJitContext *ctx, const char *symbol)
    {
        if (!ctx || !symbol)
        {
            llvm::errs() << "crab::jit: invalid arguments to crab_jit_lookup\n";
            return nullptr;
        }

        // first find directly
        auto it = ctx->symbols.find(symbol);
        if (it != ctx->symbols.end())
        {
            return it->second;
        }

        auto addr = ctx->session->lookup(symbol);
        if (!addr)
        {
            llvm::consumeError(addr.takeError());
            llvm::errs() << "crab::jit: lookup failed for symbol: " << symbol << "\n";
            return nullptr;
        }
        void *ptr = addr->toPtr<void *>();
        ctx->symbols[symbol] = ptr;
        return ptr;
    }

}