#include <Eigen/Core>
#include <utility>

// A universal helper that casts a raw void* to the exact reference type a function expects
template <typename T>
decltype(auto) unpack_arg(void* ptr) {
    if constexpr (std::is_pointer_v<T>) {
        // If the function wants a pointer (like std::array* const)
        return reinterpret_cast<T>(ptr);
    } else {
        // If the function wants a value or reference (like const Eigen::Matrix&)
        return *reinterpret_cast<std::add_pointer_t<std::remove_reference_t<T>>>(ptr);
    }
}

// The Universal Invoker: Takes any callable, unpacks raw pointer arrays, and runs it
template <typename Func, typename... Args, size_t... Is>
void invoke_unpacked(Func&& func, void** inputs, void** outputs, std::index_sequence<Is...>) {
    // This deduces the exact types expected by the SymForce function arguments
    using Traits = tuple_element_t; // (Using standard template traits to extract argument types)
    
    // It automatically casts each void* from your input/output arrays to exactly what SymForce wants
    std::forward<Func>(func)(unpack_arg<Args>(inputs[Is])...); 
}