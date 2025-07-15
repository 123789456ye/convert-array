use std::any::Any;
use std::sync::Arc;
use std::collections::HashMap;

use arrow::array::{ArrayRef, ListArray, MapArray, StructArray};
use arrow::datatypes::*;
use arrow::buffer::OffsetBuffer;

use crate::array::primitive_array::Decimal128Value;
use crate::array::time_array::*;
use crate::datatype::DynScalar;

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

/// Dispatch Any data to appropriate DynScalar type based on Arrow DataType.
pub fn convert_dyn_scalar(data: &dyn Any, data_type: &DataType) -> DynScalar {
    match data_type {
        DataType::Boolean => DynScalar::Bool(*data.downcast_ref::<bool>().unwrap()),
        DataType::Int8 => DynScalar::Int8(*data.downcast_ref::<i8>().unwrap()),
        DataType::Int16 => DynScalar::Int16(*data.downcast_ref::<i16>().unwrap()),
        DataType::Int32 => DynScalar::Int32(*data.downcast_ref::<i32>().unwrap()),
        DataType::Int64 => DynScalar::Int64(*data.downcast_ref::<i64>().unwrap()),
        DataType::UInt8 => DynScalar::UInt8(*data.downcast_ref::<u8>().unwrap()),
        DataType::UInt16 => DynScalar::UInt16(*data.downcast_ref::<u16>().unwrap()),
        DataType::UInt32 => DynScalar::UInt32(*data.downcast_ref::<u32>().unwrap()),
        DataType::UInt64 => DynScalar::UInt64(*data.downcast_ref::<u64>().unwrap()),
        DataType::Float32 => DynScalar::Float32(*data.downcast_ref::<f32>().unwrap()),
        DataType::Float64 => DynScalar::Float64(*data.downcast_ref::<f64>().unwrap()),
        DataType::Utf8 => {
            // Handle both String and Option<String>
            if let Some(s) = data.downcast_ref::<String>() {
                DynScalar::String(s.clone())
            } else if let Some(opt_s) = data.downcast_ref::<Option<String>>() {
                DynScalar::OptionalString(opt_s.clone())
            } else {
                panic!("Failed to downcast to String or Option<String>");
            }
        },
        DataType::Binary => DynScalar::Binary(data.downcast_ref::<Vec<u8>>().unwrap().clone()),
        DataType::List(_) => DynScalar::List(data.downcast_ref::<Vec<DynScalar>>().unwrap().clone()),
        DataType::Struct(_) => DynScalar::Struct(data.downcast_ref::<HashMap<String, DynScalar>>().unwrap().clone()),
        DataType::Map(_, _) => DynScalar::Map(data.downcast_ref::<Vec<(DynScalar, DynScalar)>>().unwrap().clone()),
        DataType::FixedSizeList(_, size) => DynScalar::FixedSizeList(data.downcast_ref::<Vec<DynScalar>>().unwrap().clone(), *size),
        DataType::Timestamp(unit, _) => {
            match unit {
                TimeUnit::Second => DynScalar::TimestampSecond(TimestampSecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Millisecond => DynScalar::TimestampMillisecond(TimestampMillisecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Microsecond => DynScalar::TimestampMicrosecond(TimestampMicrosecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Nanosecond => DynScalar::TimestampNanosecond(TimestampNanosecond(*data.downcast_ref::<i64>().unwrap())),
            }
        },
        DataType::Time32(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::Time32Second(Time32Second(*data.downcast_ref::<i32>().unwrap())),
                TimeUnit::Millisecond => DynScalar::Time32Millisecond(Time32Millisecond(*data.downcast_ref::<i32>().unwrap())),
                _ => panic!("Unsupported Time32 unit: {:?}", unit),
            }
        },
        DataType::Time64(unit) => {
            match unit {
                TimeUnit::Microsecond => DynScalar::Time64Microsecond(Time64Microsecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Nanosecond => DynScalar::Time64Nanosecond(Time64Nanosecond(*data.downcast_ref::<i64>().unwrap())),
                _ => panic!("Unsupported Time64 unit: {:?}", unit),
            }
        },
        DataType::Duration(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::DurationSecond(DurationSecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Millisecond => DynScalar::DurationMillisecond(DurationMillisecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Microsecond => DynScalar::DurationMicrosecond(DurationMicrosecond(*data.downcast_ref::<i64>().unwrap())),
                TimeUnit::Nanosecond => DynScalar::DurationNanosecond(DurationNanosecond(*data.downcast_ref::<i64>().unwrap())),
            }
        },
        DataType::Decimal128(precision, scale) => {
            DynScalar::Decimal128(Decimal128Value {
                value: *data.downcast_ref::<i128>().unwrap(),
                precision: *precision,
                scale: *scale,
            })
        },
        _ => panic!("Unsupported data type for default: {:?}", data_type),
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
        DataType::FixedSizeList(_, size) => DynScalar::FixedSizeList(Vec::new(), *size),
        DataType::Timestamp(unit, _) => {
            match unit {
                TimeUnit::Second => DynScalar::TimestampSecond(TimestampSecond(0)),
                TimeUnit::Millisecond => DynScalar::TimestampMillisecond(TimestampMillisecond(0)),
                TimeUnit::Microsecond => DynScalar::TimestampMicrosecond(TimestampMicrosecond(0)),
                TimeUnit::Nanosecond => DynScalar::TimestampNanosecond(TimestampNanosecond(0)),
            }
        },
        DataType::Time32(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::Time32Second(Time32Second(0)),
                TimeUnit::Millisecond => DynScalar::Time32Millisecond(Time32Millisecond(0)),
                _ => panic!("Unsupported Time32 unit: {:?}", unit),
            }
        },
        DataType::Time64(unit) => {
            match unit {
                TimeUnit::Microsecond => DynScalar::Time64Microsecond(Time64Microsecond(0)),
                TimeUnit::Nanosecond => DynScalar::Time64Nanosecond(Time64Nanosecond(0)),
                _ => panic!("Unsupported Time64 unit: {:?}", unit),
            }
        },
        DataType::Duration(unit) => {
            match unit {
                TimeUnit::Second => DynScalar::DurationSecond(DurationSecond(0)),
                TimeUnit::Millisecond => DynScalar::DurationMillisecond(DurationMillisecond(0)),
                TimeUnit::Microsecond => DynScalar::DurationMicrosecond(DurationMicrosecond(0)),
                TimeUnit::Nanosecond => DynScalar::DurationNanosecond(DurationNanosecond(0)),
            }
        },
        DataType::Decimal128(precision, scale) => {
            DynScalar::Decimal128(Decimal128Value {
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
    use arrow::array::{Array};
    use arrow::datatypes::{DataType};

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