use std::sync::Arc;
use std::collections::HashMap;

use arrow::array::{ArrayRef, ListArray, MapArray, StructArray};
use arrow::datatypes::*;
use arrow::buffer::OffsetBuffer;

use crate::datatype::DynScalar;

// Macro to generate dictionary builders for all key types
macro_rules! build_dictionary_array {
    ($key_type:ty, $values:expr) => {{
        let mut builder = arrow::array::StringDictionaryBuilder::<$key_type>::new();
        for value in $values {
            match value {
                DynScalar::Dictionary(_, value_scalar) => {
                    match value_scalar.as_ref() {
                        DynScalar::String(s) => builder.append_value(s),
                        _ => builder.append_null(),
                    }
                }
                DynScalar::Null => builder.append_null(),
                _ => panic!("Type mismatch: expected Dictionary or Null, got {:?}", value),
            }
        }
        Arc::new(builder.finish())
    }};
}

/// Converts a Vec<DynScalar> to an Arrow ArrayRef using Arrow builders for optimal performance.
/// This function uses builders consistently across all data types for better efficiency.
pub fn dyn_scalar_vec_to_array(values: Vec<DynScalar>, data_type: &DataType) -> ArrayRef {
    match data_type {
        // Primitive types using builders - handle both regular and optional variants
        DataType::Boolean => {
            let mut builder = arrow::array::BooleanBuilder::new();
            for value in values {
                match value {
                    DynScalar::Bool(v) => builder.append_value(v),
                    DynScalar::OptionalBool(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalBool(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Bool, OptionalBool, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Int8 => {
            let mut builder = arrow::array::Int8Builder::new();
            for value in values {
                match value {
                    DynScalar::Int8(v) => builder.append_value(v),
                    DynScalar::OptionalInt8(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalInt8(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Int8, OptionalInt8, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Int16 => {
            let mut builder = arrow::array::Int16Builder::new();
            for value in values {
                match value {
                    DynScalar::Int16(v) => builder.append_value(v),
                    DynScalar::OptionalInt16(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalInt16(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Int16, OptionalInt16, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Int32 => {
            let mut builder = arrow::array::Int32Builder::new();
            for value in values {
                match value {
                    DynScalar::Int32(v) => builder.append_value(v),
                    DynScalar::OptionalInt32(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalInt32(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Int32, OptionalInt32, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Int64 => {
            let mut builder = arrow::array::Int64Builder::new();
            for value in values {
                match value {
                    DynScalar::Int64(v) => builder.append_value(v),
                    DynScalar::OptionalInt64(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalInt64(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Int64, OptionalInt64, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::UInt8 => {
            let mut builder = arrow::array::UInt8Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt8(v) => builder.append_value(v),
                    DynScalar::OptionalUInt8(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalUInt8(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected UInt8, OptionalUInt8, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::UInt16 => {
            let mut builder = arrow::array::UInt16Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt16(v) => builder.append_value(v),
                    DynScalar::OptionalUInt16(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalUInt16(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected UInt16, OptionalUInt16, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::UInt32 => {
            let mut builder = arrow::array::UInt32Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt32(v) => builder.append_value(v),
                    DynScalar::OptionalUInt32(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalUInt32(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected UInt32, OptionalUInt32, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::UInt64 => {
            let mut builder = arrow::array::UInt64Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt64(v) => builder.append_value(v),
                    DynScalar::OptionalUInt64(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalUInt64(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected UInt64, OptionalUInt64, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Float32 => {
            let mut builder = arrow::array::Float32Builder::new();
            for value in values {
                match value {
                    DynScalar::Float32(v) => builder.append_value(v),
                    DynScalar::OptionalFloat32(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalFloat32(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Float32, OptionalFloat32, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Float64 => {
            let mut builder = arrow::array::Float64Builder::new();
            for value in values {
                match value {
                    DynScalar::Float64(v) => builder.append_value(v),
                    DynScalar::OptionalFloat64(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalFloat64(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Float64, OptionalFloat64, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        DataType::Utf8 => {
            let mut builder = arrow::array::StringBuilder::new();
            for value in values {
                match value {
                    DynScalar::String(v) => builder.append_value(v),
                    DynScalar::OptionalString(Some(v)) => builder.append_value(v),
                    DynScalar::OptionalString(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected String, OptionalString, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        },
        
        DataType::Binary => {
            let mut builder = arrow::array::BinaryBuilder::new();
            for value in values {
                match value {
                    DynScalar::Binary(v) => builder.append_value(&v),
                    DynScalar::OptionalBinary(Some(v)) => builder.append_value(&v),
                    DynScalar::OptionalBinary(None) | DynScalar::Null => builder.append_null(),
                    _ => panic!("Type mismatch: expected Binary, OptionalBinary, or Null, got {:?}", value),
                }
            }
            Arc::new(builder.finish())
        }
        
        // Dictionary types using optimized builders
        DataType::Dictionary(key_type, value_type) => {
            match (key_type.as_ref(), value_type.as_ref()) {
                (DataType::Int8, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::Int8Type, values),
                (DataType::Int16, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::Int16Type, values),
                (DataType::Int32, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::Int32Type, values),
                (DataType::Int64, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::Int64Type, values),
                (DataType::UInt8, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::UInt8Type, values),
                (DataType::UInt16, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::UInt16Type, values),
                (DataType::UInt32, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::UInt32Type, values),
                (DataType::UInt64, DataType::Utf8) => build_dictionary_array!(arrow::datatypes::UInt64Type, values),
                _ => {
                    // Fallback for non-string dictionary value types
                    // Use manual construction with builders for better null handling
                    let mut keys_vec = Vec::new();
                    let mut null_mask = Vec::new();
                    let mut values_set = std::collections::HashMap::new();
                    let mut values_vec = Vec::new();
                    let mut next_key_idx = 0;
                    
                    for value in &values {
                        match value {
                            DynScalar::Dictionary(_, value_scalar) => {
                                let value_str = format!("{:?}", value_scalar);
                                if !values_set.contains_key(&value_str) {
                                    values_set.insert(value_str.clone(), next_key_idx);
                                    values_vec.push(value_scalar.as_ref().clone());
                                    next_key_idx += 1;
                                }
                                let key_idx = values_set[&value_str];
                                keys_vec.push(key_idx);
                                null_mask.push(true);
                            }
                            DynScalar::Null => {
                                keys_vec.push(0);
                                null_mask.push(false);
                            }
                            _ => panic!("Type mismatch: expected Dictionary or Null, got {:?}", value),
                        }
                    }
                    
                    let values_array = dyn_scalar_vec_to_array(values_vec, value_type);
                    let validity = if null_mask.iter().all(|&x| x) {
                        None
                    } else {
                        Some(arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(null_mask)))
                    };
                    
                    match key_type.as_ref() {
                        DataType::Int8 => {
                            let keys: Vec<i8> = keys_vec.iter().map(|&k| k as i8).collect();
                            let keys_array = arrow::array::Int8Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::Int16 => {
                            let keys: Vec<i16> = keys_vec.iter().map(|&k| k as i16).collect();
                            let keys_array = arrow::array::Int16Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::Int32 => {
                            let keys: Vec<i32> = keys_vec.iter().map(|&k| k as i32).collect();
                            let keys_array = arrow::array::Int32Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::Int64 => {
                            let keys: Vec<i64> = keys_vec.iter().map(|&k| k as i64).collect();
                            let keys_array = arrow::array::Int64Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::UInt8 => {
                            let keys: Vec<u8> = keys_vec.iter().map(|&k| k as u8).collect();
                            let keys_array = arrow::array::UInt8Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::UInt16 => {
                            let keys: Vec<u16> = keys_vec.iter().map(|&k| k as u16).collect();
                            let keys_array = arrow::array::UInt16Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::UInt32 => {
                            let keys: Vec<u32> = keys_vec.iter().map(|&k| k as u32).collect();
                            let keys_array = arrow::array::UInt32Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        DataType::UInt64 => {
                            let keys: Vec<u64> = keys_vec.iter().map(|&k| k as u64).collect();
                            let keys_array = arrow::array::UInt64Array::new(keys.into(), validity);
                            let dict_array = arrow::array::DictionaryArray::try_new(keys_array, values_array).unwrap();
                            Arc::new(dict_array)
                        }
                        _ => panic!("Unsupported dictionary key type: {:?}", key_type),
                    }
                }
            }
        }
        
        // Complex nested types using builders where possible
        DataType::List(field) => {
            // Use manual construction to preserve exact field schema
            let mut offsets = vec![0i32];
            let mut list_values = Vec::new();
            let mut null_mask = Vec::new();
            
            for value in values {
                match value {
                    DynScalar::List(items) => {
                        offsets.push(offsets.last().unwrap() + items.len() as i32);
                        list_values.extend(items);
                        null_mask.push(true);
                    }
                    DynScalar::OptionalList(Some(items)) => {
                        offsets.push(offsets.last().unwrap() + items.len() as i32);
                        list_values.extend(items);
                        null_mask.push(true);
                    }
                    DynScalar::OptionalList(None) | DynScalar::Null => {
                        offsets.push(*offsets.last().unwrap());
                        null_mask.push(false);
                    }
                    _ => panic!("Type mismatch: expected List, OptionalList, or Null, got {:?}", value),
                }
            }
            
            let values_array = dyn_scalar_vec_to_array(list_values, field.data_type());
            let offsets_buffer = OffsetBuffer::new(offsets.into());
            
            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(null_mask)))
            };
            
            Arc::new(ListArray::new(
                field.clone(),
                offsets_buffer,
                values_array,
                validity,
            ))
        }
        
        DataType::Struct(fields) => {
            // Use column-wise approach with builders for each field
            let mut field_arrays: Vec<ArrayRef> = Vec::new();
            let mut null_mask = Vec::new();
            
            for field in fields {
                let field_name = field.name();
                let field_values: Vec<DynScalar> = values.iter()
                    .map(|v| match v {
                        DynScalar::Struct(map) => {
                            map.get(field_name).cloned().unwrap_or_else(|| default_dyn_scalar(field.data_type()))
                        }
                        DynScalar::OptionalStruct(Some(map)) => {
                            map.get(field_name).cloned().unwrap_or_else(|| default_dyn_scalar(field.data_type()))
                        }
                        DynScalar::OptionalStruct(None) | DynScalar::Null => DynScalar::Null,
                        _ => panic!("Type mismatch: expected Struct, OptionalStruct, or Null, got {:?}", v),
                    })
                    .collect();
                
                let array = dyn_scalar_vec_to_array(field_values, field.data_type());
                field_arrays.push(array);
            }
            
            // Create null mask for struct level nulls
            for value in &values {
                match value {
                    DynScalar::Struct(_) | DynScalar::OptionalStruct(Some(_)) => {
                        null_mask.push(true);
                    }
                    DynScalar::OptionalStruct(None) | DynScalar::Null => {
                        null_mask.push(false);
                    }
                    _ => panic!("Type mismatch: expected Struct, OptionalStruct, or Null, got {:?}", value),
                }
            }
            
            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(null_mask)))
            };
            
            Arc::new(StructArray::new(
                fields.clone(),
                field_arrays,
                validity,
            ))
        }
        
        DataType::Map(field, _) => {
            // Use manual construction for Maps - complex type that doesn't have simple builder
            let struct_fields = match field.data_type() {
                DataType::Struct(fields) => fields,
                _ => panic!("Map field must contain a struct type"),
            };
            
            if struct_fields.len() != 2 {
                panic!("Map struct must have exactly 2 fields (key and value)");
            }
            
            let key_field = &struct_fields[0];
            let value_field = &struct_fields[1];
            
            let mut offsets = vec![0i32];
            let mut keys = Vec::new();
            let mut map_values = Vec::new();
            let mut null_mask = Vec::new();
            
            for value in values {
                match value {
                    DynScalar::Map(pairs) => {
                        offsets.push(offsets.last().unwrap() + pairs.len() as i32);
                        for (k, v) in pairs {
                            keys.push(k);
                            map_values.push(v);
                        }
                        null_mask.push(true);
                    }
                    DynScalar::OptionalMap(Some(pairs)) => {
                        offsets.push(offsets.last().unwrap() + pairs.len() as i32);
                        for (k, v) in pairs {
                            keys.push(k);
                            map_values.push(v);
                        }
                        null_mask.push(true);
                    }
                    DynScalar::OptionalMap(None) | DynScalar::Null => {
                        offsets.push(*offsets.last().unwrap());
                        null_mask.push(false);
                    }
                    _ => panic!("Type mismatch: expected Map, OptionalMap, or Null, got {:?}", value),
                }
            }
            
            let keys_array = dyn_scalar_vec_to_array(keys, key_field.data_type());
            let values_array = dyn_scalar_vec_to_array(map_values, value_field.data_type());
            let offsets_buffer = OffsetBuffer::new(offsets.into());
            
            let struct_array = StructArray::new(
                struct_fields.clone(),
                vec![keys_array, values_array],
                None,
            );
            
            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(null_mask)))
            };
            
            Arc::new(MapArray::new(
                field.clone(),
                offsets_buffer,
                struct_array,
                validity,
                false,
            ))
        }
        
        DataType::FixedSizeList(field, size) => {
            let mut flat_values = Vec::new();
            let mut null_mask = Vec::new();
            
            for value in values {
                match value {
                    DynScalar::FixedSizeList(items, _list_size) => {
                        if items.len() != *size as usize {
                            panic!("FixedSizeList size mismatch: expected {}, got {}", size, items.len());
                        }
                        flat_values.extend(items);
                        null_mask.push(true);
                    }
                    DynScalar::OptionalFixedSizeList(Some(items), _list_size) => {
                        if items.len() != *size as usize {
                            panic!("FixedSizeList size mismatch: expected {}, got {}", size, items.len());
                        }
                        flat_values.extend(items);
                        null_mask.push(true);
                    }
                    DynScalar::OptionalFixedSizeList(None, _) | DynScalar::Null => {
                        for _ in 0..*size {
                            flat_values.push(DynScalar::Null);
                        }
                        null_mask.push(false);
                    }
                    _ => panic!("Type mismatch: expected FixedSizeList, OptionalFixedSizeList, or Null, got {:?}", value),
                }
            }
            
            let values_array = dyn_scalar_vec_to_array(flat_values, field.data_type());
            
            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(arrow::buffer::BooleanBuffer::from(null_mask)))
            };
            
            Arc::new(arrow::array::FixedSizeListArray::new(
                field.clone(),
                *size,
                values_array,
                validity,
            ))
        }
        
        _ => panic!("Unsupported data type: {:?}", data_type),
    }
}

/// Returns a default DynScalar value for the given Arrow DataType.
/// Used for filling missing values in struct fields.
pub fn default_dyn_scalar(data_type: &DataType) -> DynScalar {
    match data_type {
        DataType::Boolean => DynScalar::Bool(false),
        DataType::Int8 => DynScalar::Int8(0),
        DataType::Int16 => DynScalar::Int16(0),
        DataType::Int32 => DynScalar::Int32(0),
        DataType::Int64 => DynScalar::Int64(0),
        DataType::UInt8 => DynScalar::UInt8(0),
        DataType::UInt16 => DynScalar::UInt16(0),
        DataType::UInt32 => DynScalar::UInt32(0),
        DataType::UInt64 => DynScalar::UInt64(0),
        DataType::Float32 => DynScalar::Float32(0.0),
        DataType::Float64 => DynScalar::Float64(0.0),
        DataType::Utf8 => DynScalar::String(String::new()),
        DataType::Binary => DynScalar::Binary(Vec::new()),
        DataType::List(_) => DynScalar::List(Vec::new()),
        DataType::Struct(_) => DynScalar::Struct(HashMap::new()),
        DataType::Map(_, _) => DynScalar::Map(Vec::new()),
        DataType::Dictionary(key_type, value_type) => {
            let default_key = match key_type.as_ref() {
                DataType::Int8 => DynScalar::Int8(0),
                DataType::Int16 => DynScalar::Int16(0),
                DataType::Int32 => DynScalar::Int32(0),
                DataType::Int64 => DynScalar::Int64(0),
                DataType::UInt8 => DynScalar::UInt8(0),
                DataType::UInt16 => DynScalar::UInt16(0),
                DataType::UInt32 => DynScalar::UInt32(0),
                DataType::UInt64 => DynScalar::UInt64(0),
                _ => panic!("Unsupported dictionary key type: {:?}", key_type),
            };
            let default_value = default_dyn_scalar(value_type);
            DynScalar::Dictionary(Box::new(default_key), Box::new(default_value))
        },
        DataType::FixedSizeList(_, size) => DynScalar::FixedSizeList(Vec::new(), *size),
        DataType::Timestamp(unit, _) => {
            match unit {
                TimeUnit::Second => DynScalar::TimestampSecond(crate::array::time_array::TimestampSecond(0)),
                TimeUnit::Millisecond => DynScalar::TimestampMillisecond(crate::array::time_array::TimestampMillisecond(0)),
                TimeUnit::Microsecond => DynScalar::TimestampMicrosecond(crate::array::time_array::TimestampMicrosecond(0)),
                TimeUnit::Nanosecond => DynScalar::TimestampNanosecond(crate::array::time_array::TimestampNanosecond(0)),
            }
        },
        DataType::Time32(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::Time32Second(crate::array::time_array::Time32Second(0)),
                TimeUnit::Millisecond => DynScalar::Time32Millisecond(crate::array::time_array::Time32Millisecond(0)),
                _ => panic!("Unsupported Time32 unit: {:?}", unit),
            }
        },
        DataType::Time64(unit) => {
            match unit {
                TimeUnit::Microsecond => DynScalar::Time64Microsecond(crate::array::time_array::Time64Microsecond(0)),
                TimeUnit::Nanosecond => DynScalar::Time64Nanosecond(crate::array::time_array::Time64Nanosecond(0)),
                _ => panic!("Unsupported Time64 unit: {:?}", unit),
            }
        },
        DataType::Duration(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::DurationSecond(crate::array::time_array::DurationSecond(0)),
                TimeUnit::Millisecond => DynScalar::DurationMillisecond(crate::array::time_array::DurationMillisecond(0)),
                TimeUnit::Microsecond => DynScalar::DurationMicrosecond(crate::array::time_array::DurationMicrosecond(0)),
                TimeUnit::Nanosecond => DynScalar::DurationNanosecond(crate::array::time_array::DurationNanosecond(0)),
            }
        },
        DataType::Decimal128(precision, scale) => {
            DynScalar::Decimal128(crate::array::time_array::Decimal128Value {
                value: 0,
                precision: *precision,
                scale: *scale,
            })
        },
        _ => panic!("Unsupported data type for default: {:?}", data_type),
    }
}

/// Returns a default optional DynScalar value for the given Arrow DataType.
/// Used for creating optional variants with None values.
pub fn default_optional_dyn_scalar(data_type: &DataType) -> DynScalar {
    match data_type {
        DataType::Boolean => DynScalar::OptionalBool(None),
        DataType::Int8 => DynScalar::OptionalInt8(None),
        DataType::Int16 => DynScalar::OptionalInt16(None),
        DataType::Int32 => DynScalar::OptionalInt32(None),
        DataType::Int64 => DynScalar::OptionalInt64(None),
        DataType::UInt8 => DynScalar::OptionalUInt8(None),
        DataType::UInt16 => DynScalar::OptionalUInt16(None),
        DataType::UInt32 => DynScalar::OptionalUInt32(None),
        DataType::UInt64 => DynScalar::OptionalUInt64(None),
        DataType::Float32 => DynScalar::OptionalFloat32(None),
        DataType::Float64 => DynScalar::OptionalFloat64(None),
        DataType::Utf8 => DynScalar::OptionalString(None),
        DataType::Binary => DynScalar::OptionalBinary(None),
        DataType::List(_) => DynScalar::OptionalList(None),
        DataType::Struct(_) => DynScalar::OptionalStruct(None),
        DataType::Map(_, _) => DynScalar::OptionalMap(None),
        DataType::Dictionary(_, _) => DynScalar::OptionalDictionary(None, None),
        DataType::FixedSizeList(_, size) => DynScalar::OptionalFixedSizeList(None, *size),
        DataType::Timestamp(unit, _) => {
            match unit {
                TimeUnit::Second => DynScalar::OptionalTimestampSecond(None),
                TimeUnit::Millisecond => DynScalar::OptionalTimestampMillisecond(None),
                TimeUnit::Microsecond => DynScalar::OptionalTimestampMicrosecond(None),
                TimeUnit::Nanosecond => DynScalar::OptionalTimestampNanosecond(None),
            }
        },
        DataType::Time32(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::OptionalTime32Second(None),
                TimeUnit::Millisecond => DynScalar::OptionalTime32Millisecond(None),
                _ => panic!("Unsupported Time32 unit: {:?}", unit),
            }
        },
        DataType::Time64(unit) => {
            match unit {
                TimeUnit::Microsecond => DynScalar::OptionalTime64Microsecond(None),
                TimeUnit::Nanosecond => DynScalar::OptionalTime64Nanosecond(None),
                _ => panic!("Unsupported Time64 unit: {:?}", unit),
            }
        },
        DataType::Duration(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::OptionalDurationSecond(None),
                TimeUnit::Millisecond => DynScalar::OptionalDurationMillisecond(None),
                TimeUnit::Microsecond => DynScalar::OptionalDurationMicrosecond(None),
                TimeUnit::Nanosecond => DynScalar::OptionalDurationNanosecond(None),
            }
        },
        DataType::Decimal128(_, _) => DynScalar::OptionalDecimal128(None),
        _ => panic!("Unsupported data type for optional default: {:?}", data_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, DictionaryArray, StringArray, StringDictionaryBuilder};
    use arrow::datatypes::{DataType, Int32Type};

    #[test]
    fn test_dictionary_array_vs_builder() {
        // Test data: ["a", "a", "b", "c", "a"]
        let input_values = vec![
            DynScalar::Dictionary(Box::new(DynScalar::Int32(0)), Box::new(DynScalar::String("a".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int32(0)), Box::new(DynScalar::String("a".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int32(1)), Box::new(DynScalar::String("b".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int32(2)), Box::new(DynScalar::String("c".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int32(0)), Box::new(DynScalar::String("a".to_string()))),
        ];

        let data_type = DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        );

        // Create array using our implementation
        let our_array = dyn_scalar_vec_to_array(input_values, &data_type);
        
        // Create array using Arrow builder
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        builder.append_value("a");
        builder.append_value("a");
        builder.append_value("b");
        builder.append_value("c");
        builder.append_value("a");
        let expected_array = builder.finish();

        // Compare the arrays
        let our_dict = our_array.as_any().downcast_ref::<DictionaryArray<Int32Type>>().unwrap();
        let expected_dict = &expected_array;

        // Compare keys
        assert_eq!(our_dict.keys().len(), expected_dict.keys().len());
        
        // Compare values (dictionary)
        let our_values = our_dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        let expected_values = expected_dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        
        assert_eq!(*our_values, *expected_values);
        // The values arrays should contain the same unique strings (order might differ)
        let our_values_set: std::collections::HashSet<_> = (0..our_values.len()).map(|i| our_values.value(i)).collect();
        let expected_values_set: std::collections::HashSet<_> = (0..expected_values.len()).map(|i| expected_values.value(i)).collect();
        assert_eq!(our_values_set, expected_values_set);

        // Compare logical values by iterating through both arrays
        let our_logical: Vec<_> = our_dict.downcast_dict::<StringArray>().unwrap().into_iter().collect();
        let expected_logical: Vec<_> = expected_dict.downcast_dict::<StringArray>().unwrap().into_iter().collect();
        
        assert_eq!(our_logical, expected_logical);
        assert_eq!(our_logical, vec![Some("a"), Some("a"), Some("b"), Some("c"), Some("a")]);
    }

    #[test]
    fn test_dictionary_array_with_nulls() {
        // Test data: ["a", null, "b", "a", null]
        let input_values = vec![
            DynScalar::Dictionary(Box::new(DynScalar::Int32(0)), Box::new(DynScalar::String("a".to_string()))),
            DynScalar::Null,
            DynScalar::Dictionary(Box::new(DynScalar::Int32(1)), Box::new(DynScalar::String("b".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int32(0)), Box::new(DynScalar::String("a".to_string()))),
            DynScalar::Null,
        ];

        let data_type = DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        );

        // Create array using our implementation
        let our_array = dyn_scalar_vec_to_array(input_values, &data_type);
        
        // Create array using Arrow builder
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        builder.append_value("a");
        builder.append_null();
        builder.append_value("b");
        builder.append_value("a");
        builder.append_null();
        let expected_array = builder.finish();

        // Compare logical values
        let our_dict = our_array.as_any().downcast_ref::<DictionaryArray<Int32Type>>().unwrap();
        let expected_dict = &expected_array;

        let our_logical: Vec<_> = our_dict.downcast_dict::<StringArray>().unwrap().into_iter().collect();
        let expected_logical: Vec<_> = expected_dict.downcast_dict::<StringArray>().unwrap().into_iter().collect();
        
        assert_eq!(our_logical, expected_logical);
        assert_eq!(our_logical, vec![Some("a"), None, Some("b"), Some("a"), None]);
    }

    #[test]
    fn test_dictionary_array_different_key_types() {
        // Test with Int8 keys
        let input_values = vec![
            DynScalar::Dictionary(Box::new(DynScalar::Int8(0)), Box::new(DynScalar::String("x".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int8(0)), Box::new(DynScalar::String("x".to_string()))),
            DynScalar::Dictionary(Box::new(DynScalar::Int8(1)), Box::new(DynScalar::String("y".to_string()))),
        ];

        let data_type = DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(DataType::Utf8),
        );

        let our_array = dyn_scalar_vec_to_array(input_values, &data_type);
        let our_dict = our_array.as_any().downcast_ref::<DictionaryArray<arrow::datatypes::Int8Type>>().unwrap();
        
        let logical_values: Vec<_> = our_dict.downcast_dict::<StringArray>().unwrap().into_iter().collect();
        assert_eq!(logical_values, vec![Some("x"), Some("x"), Some("y")]);
        
        // Verify key type
        assert_eq!(our_dict.keys().data_type(), &DataType::Int8);
    }

    #[test]
    fn test_optional_primitive_types() {
        // Test OptionalInt32 with mixed Some/None values
        let values = vec![
            DynScalar::OptionalInt32(Some(42)),
            DynScalar::OptionalInt32(None),
            DynScalar::OptionalInt32(Some(100)),
            DynScalar::Null,
        ];
        let data_type = DataType::Int32;
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let int32_array = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
        assert_eq!(int32_array.len(), 4);
        assert_eq!(int32_array.value(0), 42);
        assert!(int32_array.is_null(1));
        assert_eq!(int32_array.value(2), 100);
        assert!(int32_array.is_null(3));
    }

    #[test]
    fn test_optional_string_types() {
        // Test OptionalString with mixed Some/None values
        let values = vec![
            DynScalar::OptionalString(Some("hello".to_string())),
            DynScalar::OptionalString(None),
            DynScalar::OptionalString(Some("world".to_string())),
            DynScalar::Null,
        ];
        let data_type = DataType::Utf8;
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let string_array = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        assert_eq!(string_array.len(), 4);
        assert_eq!(string_array.value(0), "hello");
        assert!(string_array.is_null(1));
        assert_eq!(string_array.value(2), "world");
        assert!(string_array.is_null(3));
    }

    #[test]
    fn test_optional_bool_types() {
        // Test OptionalBool with mixed Some/None values
        let values = vec![
            DynScalar::OptionalBool(Some(true)),
            DynScalar::OptionalBool(None),
            DynScalar::OptionalBool(Some(false)),
            DynScalar::Null,
        ];
        let data_type = DataType::Boolean;
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let bool_array = array.as_any().downcast_ref::<arrow::array::BooleanArray>().unwrap();
        assert_eq!(bool_array.len(), 4);
        assert_eq!(bool_array.value(0), true);
        assert!(bool_array.is_null(1));
        assert_eq!(bool_array.value(2), false);
        assert!(bool_array.is_null(3));
    }

    #[test]
    fn test_optional_binary_types() {
        // Test OptionalBinary with mixed Some/None values
        let values = vec![
            DynScalar::OptionalBinary(Some(vec![1, 2, 3])),
            DynScalar::OptionalBinary(None),
            DynScalar::OptionalBinary(Some(vec![4, 5, 6])),
            DynScalar::Null,
        ];
        let data_type = DataType::Binary;
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let binary_array = array.as_any().downcast_ref::<arrow::array::BinaryArray>().unwrap();
        assert_eq!(binary_array.len(), 4);
        assert_eq!(binary_array.value(0), &[1, 2, 3]);
        assert!(binary_array.is_null(1));
        assert_eq!(binary_array.value(2), &[4, 5, 6]);
        assert!(binary_array.is_null(3));
    }

    #[test]
    fn test_mixed_regular_and_optional_types() {
        // Test mixing regular and optional variants
        let values = vec![
            DynScalar::Int32(42),
            DynScalar::OptionalInt32(Some(100)),
            DynScalar::OptionalInt32(None),
            DynScalar::Null,
            DynScalar::Int32(200),
        ];
        let data_type = DataType::Int32;
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let int32_array = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
        assert_eq!(int32_array.len(), 5);
        assert_eq!(int32_array.value(0), 42);
        assert_eq!(int32_array.value(1), 100);
        assert!(int32_array.is_null(2));
        assert!(int32_array.is_null(3));
        assert_eq!(int32_array.value(4), 200);
    }

    #[test]
    fn test_optional_nested_types() {
        // Test OptionalList with mixed Some/None values
        let values = vec![
            DynScalar::OptionalList(Some(vec![
                DynScalar::Int32(1),
                DynScalar::Int32(2),
            ])),
            DynScalar::OptionalList(None),
            DynScalar::OptionalList(Some(vec![
                DynScalar::Int32(3),
                DynScalar::Int32(4),
                DynScalar::Int32(5),
            ])),
            DynScalar::Null,
        ];
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let data_type = DataType::List(field);
        let array = dyn_scalar_vec_to_array(values, &data_type);
        
        // Check the array contents
        let list_array = array.as_any().downcast_ref::<arrow::array::ListArray>().unwrap();
        assert_eq!(list_array.len(), 4);
        
        // Check first list: [1, 2]
        let first_list = list_array.value(0);
        let first_int_array = first_list.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
        assert_eq!(first_int_array.len(), 2);
        assert_eq!(first_int_array.value(0), 1);
        assert_eq!(first_int_array.value(1), 2);
        
        // Check second list: null
        assert!(list_array.is_null(1));
        
        // Check third list: [3, 4, 5]
        let third_list = list_array.value(2);
        let third_int_array = third_list.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
        assert_eq!(third_int_array.len(), 3);
        assert_eq!(third_int_array.value(0), 3);
        assert_eq!(third_int_array.value(1), 4);
        assert_eq!(third_int_array.value(2), 5);
        
        // Check fourth list: null
        assert!(list_array.is_null(3));
    }

    #[test]
    fn test_default_optional_dyn_scalar() {
        // Test default optional values for different types
        assert_eq!(default_optional_dyn_scalar(&DataType::Boolean), DynScalar::OptionalBool(None));
        assert_eq!(default_optional_dyn_scalar(&DataType::Int32), DynScalar::OptionalInt32(None));
        assert_eq!(default_optional_dyn_scalar(&DataType::Utf8), DynScalar::OptionalString(None));
        assert_eq!(default_optional_dyn_scalar(&DataType::Binary), DynScalar::OptionalBinary(None));
        
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let list_type = DataType::List(field);
        assert_eq!(default_optional_dyn_scalar(&list_type), DynScalar::OptionalList(None));
    }
}