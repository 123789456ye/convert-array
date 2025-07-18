# convert-array

This crate is to convert arrays that consist of Rust native types into Apache Arrow arrays.

## Code Layout and Functionality

The codebase is organized into several modules:

- **`array/`** - Core array conversion functionality
  - `primitive_array.rs` - Handles primitive types (integers, floats, booleans, strings, binary, decimals)
  - `time_array.rs` - Handles temporal types (timestamps, time, duration)
  - `nested_array.rs` - Handles nested types (lists, structs, maps, fixed-size lists)
  - `dyn_array.rs` - Dynamic array handling with type-erased operations
  - `dispatch.rs` - Dispatch to DynScalar type
- **`datatype.rs`** - Defines `DynScalar` enum for type-erased scalar values
- **`macros.rs`** - Utility macros for code generation

## Supported Types

### Primitive Types
- **Boolean**: `bool`
- **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- **Floats**: `f32`, `f64`
- **String**: `String`
- **Binary**: `Vec<u8>`
- **Decimal**: `Decimal128`

### Temporal Types
- **Timestamps**: Second, Millisecond, Microsecond, Nanosecond precision
- **Time**: 32-bit (second/millisecond), 64-bit (microsecond/nanosecond)
- **Duration**: Second, Millisecond, Microsecond, Nanosecond precision

### Nested Types
- **List**: `Vec<T>`
- **Struct**: `HashMap<String, T>`
- **Map**: `Vec<(K, V)>`
- **FixedSizeList**: Fixed-length vectors

### Optional Types
All types above support optional variants using `Option<T>`, and should use optional generally.

## Usage Examples

### When Target Type is Known at Compile Time

Use `NativeArray` implementations when you know the exact Arrow type you want at compile time:

#### Primitive Arrays
```rust
use convert_array::array::primitive_array::{TypedVec, NativeArray};
use arrow::datatypes::Int32Type;

let nullable_numbers = vec![Some(1i32), None, Some(3i32), Some(4i32)];
let int_vec = Int32Vec::from_vec(nullable_numbers);
let int32_array_with_nulls = int_vec.to_arrow_array(); // Returns Int32Array

// String data
let nullable_strings = vec![Some("hello".to_string()), None, Some("world".to_string())];
let string_vec = StringVec::from(nullable_strings);
let string_array = string_vec.to_arrow_array(); // Returns StringArray
```

#### Nested Arrays
```rust
use convert_array::array::nested_array::{ListVec, StructVec};
use arrow::datatypes::{DataType, Field, Schema};

// Non-nullable lists
let inner_field = Field::new("item", DataType::Int32, false);
let mut list_vec = ListVec::new(inner_field);
list_vec.push(vec![Some(1), Some(2), Some(3)]);
list_vec.push(vec![Some(4), Some(5)]);
let list_array = list_vec.to_arrow_array(); // Returns ListArray

// Nullable lists
let nullable_lists = vec![
    Some(vec![Some(1i32), Some(2), Some(3)]),
    None,
    Some(vec![Some(4i32), Some(5i32)]),
];
let inner_field = Field::new("item", DataType::Int32, false);
let list_vec = ListVec::from_vec_with_field(nullable_lists, inner_field);
let list_array_with_nulls = list_vec.to_arrow_array(); // Returns ListArray with nulls
```

#### Struct Arrays
```rust
use convert_array::array::nested_array::{StructVec, convert_vector_with_schema};
use convert_array::register_struct;
use arrow::datatypes::{DataType, Field, Schema};

// Define your struct
// Use proc macro to impl Into<DynScalar> automatically
#[derive(IntoDynScalar)]
struct Person {
    name: String,
    age: i32,
    email: Option<String>,
}

// Create Arrow schema
let fields = vec![
    Field::new("name", DataType::Utf8, false),
    Field::new("age", DataType::Int32, false),
    Field::new("email", DataType::Utf8, true), // nullable
];
let schema = Schema::new(fields);

// Register the struct with the schema using the macro
register_struct!(Person, schema.clone(), {
    "name" => name,
    "age" => age,
    "email" => email,
});

// Create and convert data
let people = vec![
    Person { name: "Alice".to_string(), age: 30, email: Some("alice@example.com".to_string()) },
    Person { name: "Bob".to_string(), age: 25, email: None },
];

let converted: Vec<DynScalar> = test_data.into_iter().map(|x| x.into()).collect();
let arrow_array = dynscalar_vec_to_array(converted.clone(), &DataType::Struct(fields.clone().into()));// Returns ArrayRef
```

### When Target Type is Unknown at Compile Time

Use dynamic conversion when the target Arrow type is determined at runtime:

#### Using DynNativeArray
```rust
use convert_array::array::dyn_array::ToDynArray;
use arrow::array::ArrayRef;

// Target type determined at runtime
let input = vec![Some(1i32), Some(2), Some(42)];
let dyn_array = input.to_dyn_array().unwrap();
let array_ref: ArrayRef = dyn_array.to_arrow_array(); // Returns ArrayRef (type-erased)
```

#### Using Dispatch
```rust
use convert_array::array::dispatch::dynscalar_vec_to_array;
use convert_array::datatype::DynScalar;
use arrow::datatypes::{DataType, Field};

// Start with native Rust types
let native_data = vec![
    vec![Some(1i32), Some(2)],
    vec![Some(3i32), Some(4)],
    vec![Some(5i32), Some(6), Some(7)],
];

// Convert to Vec<DynScalar> using Into implementations
let values: Vec<DynScalar> = native_data.into_iter().map(|x| x.into()).collect();

// Target type determined at runtime
let inner_field = Field::new("item", DataType::Int32, false);
let list_type = DataType::List(std::sync::Arc::new(inner_field));
let array_ref = dynscalar_vec_to_array(values, &list_type); // Returns ArrayRef
```

## Conversion Routes

The conversion process follows two fundamental approaches based on whether the target Arrow type is known at compile time:

### Route 1: Static Type Conversion (Target Type Known at Compile Time)

When you know the exact Arrow type you want, use `NativeArray` implementations(ignore Option):

**Native Rust Types → NativeArray → Concrete Arrow Arrays**

- `Vec<i32>` → `TypedVec<Int32Type>` → `Int32Array`
- `Vec<Vec<i32>>` → `ListVec` → `ListArray`
- `Vec<MyStruct>` → `StructVec` → `StructArray`
- `Vec<HashMap<K,V>>` → `MapVec` → `MapArray`

**Benefits:**
- Compile-time type safety
- Better performance (no type erasure)
- Direct access to concrete Arrow array methods

### Route 2: Dynamic Type Conversion (Target Type Unknown at Compile Time)

When the target Arrow type is determined at runtime, use dynamic conversion:

#### Route 2a: Via DynNativeArray
**Native Rust Types → DynNativeArray → ArrayRef**

- Any `Vec<T>` → `DynNativeArray<T>` → `ArrayRef` (type-erased)

#### Route 2b: Via Dispatch
**Native Rust Types → DynScalar → ArrayRef**

- Any `Vec<T>` → `Vec<DynScalar>` → `ArrayRef` (type-erased)

**Benefits:**
- Runtime type flexibility
- Unified interface for all types
- Suitable for generic/dynamic scenarios

**Special:** Though we "know" the type of Struct, we still directly convert it into DynScalar then to ArrayRef.

### Key Differences

| Aspect | Static (NativeArray) | Dynamic (DynNativeArray/Dispatch) |
|--------|---------------------|-----------------------------------|
| **Type Safety** | Compile-time | Runtime |
| **Performance** | Optimal | Slightly overhead from type erasure |
| **Return Type** | Concrete `XxxArray` | Type-erased `ArrayRef` |
| **Use Case** | Known schema at compile time | Schema determined at runtime |
| **API Access** | Direct array methods | Requires downcasting |

### When to Use Each Route

**Use Static Route when:**
- You know the exact Arrow schema at compile time
- You need maximum performance
- You want compile-time type safety
- You're working with fixed data structures

**Use Dynamic Route when:**
- Schema is determined at runtime (e.g., from config files, user input)
- You're building generic data processing pipelines
- You need to handle multiple different types uniformly
- You're working with heterogeneous data sources

The `dispatch` module uses Arrow builders for optimal performance when converting `Vec<DynScalar>` to Arrow arrays, supporting both regular and optional variants of all data types.