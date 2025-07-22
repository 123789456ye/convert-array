use std::sync::Arc;

use arrow::array::{ArrayRef, ListArray, MapArray, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::*;

use crate::datatype::DynScalar;

/// Converts a Vec<DynScalar> to an Arrow ArrayRef using Arrow builders for optimal performance.
/// This function uses builders consistently across all data types for better efficiency.
pub fn dynscalar_vec_to_array(values: Vec<DynScalar>, data_type: &DataType) -> ArrayRef {
    match data_type {
        // Primitive types using builders - handle both regular and  variants
        DataType::Boolean => {
            let mut builder = arrow::array::BooleanBuilder::new();
            for value in values {
                match value {
                    DynScalar::Bool(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Bool, Bool, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Int8 => {
            let mut builder = arrow::array::Int8Builder::new();
            for value in values {
                match value {
                    DynScalar::Int8(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Int8, Int8, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Int16 => {
            let mut builder = arrow::array::Int16Builder::new();
            for value in values {
                match value {
                    DynScalar::Int16(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Int16, Int16, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Int32 => {
            let mut builder = arrow::array::Int32Builder::new();
            for value in values {
                match value {
                    DynScalar::Int32(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Int32, Int32, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Int64 => {
            let mut builder = arrow::array::Int64Builder::new();
            for value in values {
                match value {
                    DynScalar::Int64(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Int64, Int64, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::UInt8 => {
            let mut builder = arrow::array::UInt8Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt8(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected UInt8, UInt8, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::UInt16 => {
            let mut builder = arrow::array::UInt16Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt16(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected UInt16, UInt16, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::UInt32 => {
            let mut builder = arrow::array::UInt32Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt32(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected UInt32, UInt32, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::UInt64 => {
            let mut builder = arrow::array::UInt64Builder::new();
            for value in values {
                match value {
                    DynScalar::UInt64(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected UInt64, UInt64, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Float32 => {
            let mut builder = arrow::array::Float32Builder::new();
            for value in values {
                match value {
                    DynScalar::Float32(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Float32, Float32, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Float64 => {
            let mut builder = arrow::array::Float64Builder::new();
            for value in values {
                match value {
                    DynScalar::Float64(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Float64, Float64, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }
        DataType::Utf8 => {
            let mut builder = arrow::array::StringBuilder::new();
            for value in values {
                match value {
                    DynScalar::String(v) => builder.append_value(v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected String, String, or Null, got {:?}",
                        value
                    ),
                }
            }
            Arc::new(builder.finish())
        }

        DataType::Binary => {
            let mut builder = arrow::array::BinaryBuilder::new();
            for value in values {
                match value {
                    DynScalar::Binary(v) => builder.append_value(&v),
                    DynScalar::Null => builder.append_null(),
                    _ => panic!(
                        "Type mismatch: expected Binary, Binary, or Null, got {:?}",
                        value
                    ),
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
                    DynScalar::Null => {
                        offsets.push(*offsets.last().unwrap());
                        null_mask.push(false);
                    }
                    _ => panic!(
                        "Type mismatch: expected List, List, or Null, got {:?}",
                        value
                    ),
                }
            }

            let values_array = dynscalar_vec_to_array(list_values, field.data_type());
            let offsets_buffer = OffsetBuffer::new(offsets.into());

            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(null_mask))
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
                let field_values: Vec<DynScalar> = values
                    .iter()
                    .map(|v| match v {
                        DynScalar::Struct(map) => {
                            match map.get(field_name) {
                                Some(child) => child.clone(),
                                None => DynScalar::Null 
                            }
                        },
                        DynScalar::Null => DynScalar::Null,
                        _ => panic!(
                            "Type mismatch: expected Struct, Struct, or Null, got {:?}",
                            v
                        ),
                    })
                    .collect();

                let array = dynscalar_vec_to_array(field_values, field.data_type());
                field_arrays.push(array);
            }

            // Create null mask for struct level nulls
            for value in &values {
                match value {
                    DynScalar::Struct(_) => {
                        null_mask.push(true);
                    }
                    DynScalar::Null => {
                        null_mask.push(false);
                    }
                    _ => panic!(
                        "Type mismatch: expected Struct, Struct, or Null, got {:?}",
                        value
                    ),
                }
            }

            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(null_mask))
            };

            Arc::new(StructArray::new(fields.clone(), field_arrays, validity))
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
                    DynScalar::Null => {
                        offsets.push(*offsets.last().unwrap());
                        null_mask.push(false);
                    }
                    _ => panic!("Type mismatch: expected Map, Map, or Null, got {:?}", value),
                }
            }

            let keys_array = dynscalar_vec_to_array(keys, key_field.data_type());
            let values_array = dynscalar_vec_to_array(map_values, value_field.data_type());
            let offsets_buffer = OffsetBuffer::new(offsets.into());

            let struct_array =
                StructArray::new(struct_fields.clone(), vec![keys_array, values_array], None);

            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(null_mask))
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
                            panic!(
                                "FixedSizeList size mismatch: expected {}, got {}",
                                size,
                                items.len()
                            );
                        }
                        flat_values.extend(items);
                        null_mask.push(true);
                    }
                    DynScalar::Null => {
                        for _ in 0..*size {
                            flat_values.push(DynScalar::Null);
                        }
                        null_mask.push(false);
                    }
                    _ => panic!(
                        "Type mismatch: expected FixedSizeList, FixedSizeList, or Null, got {:?}",
                        value
                    ),
                }
            }

            let values_array = dynscalar_vec_to_array(flat_values, field.data_type());

            let validity = if null_mask.iter().all(|&x| x) {
                None
            } else {
                Some(arrow::buffer::NullBuffer::from(null_mask))
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array};
    use arrow::datatypes::DataType;

    #[test]
    fn test_primitive_types() {
        // Test Int32 with mixed Some/None values
        let values = vec![
            DynScalar::Int32(42),
            DynScalar::Null,
            DynScalar::Int32(100),
            DynScalar::Null,
        ];
        let data_type = DataType::Int32;
        let array = dynscalar_vec_to_array(values, &data_type);

        // Check the array contents
        let int32_array = array
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(int32_array.len(), 4);
        assert_eq!(int32_array.value(0), 42);
        assert!(int32_array.is_null(1));
        assert_eq!(int32_array.value(2), 100);
        assert!(int32_array.is_null(3));
    }

    #[test]
    fn test_string_types() {
        // Test String with mixed /None values
        let values = vec![
            DynScalar::String("hello".to_string()),
            DynScalar::Null,
            DynScalar::String("world".to_string()),
            DynScalar::Null,
        ];
        let data_type = DataType::Utf8;
        let array = dynscalar_vec_to_array(values, &data_type);

        // Check the array contents
        let string_array = array
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(string_array.len(), 4);
        assert_eq!(string_array.value(0), "hello");
        assert!(string_array.is_null(1));
        assert_eq!(string_array.value(2), "world");
        assert!(string_array.is_null(3));
    }

    #[test]
    fn test_bool_types() {
        // Test Bool with mixed /None values
        let values = vec![
            DynScalar::Bool(true),
            DynScalar::Null,
            DynScalar::Bool(false),
            DynScalar::Null,
        ];
        let data_type = DataType::Boolean;
        let array = dynscalar_vec_to_array(values, &data_type);

        // Check the array contents
        let bool_array = array
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap();
        assert_eq!(bool_array.len(), 4);
        assert_eq!(bool_array.value(0), true);
        assert!(bool_array.is_null(1));
        assert_eq!(bool_array.value(2), false);
        assert!(bool_array.is_null(3));
    }

    #[test]
    fn test_binary_types() {
        // Test Binary with mixed /None values
        let values = vec![
            DynScalar::Binary(vec![1, 2, 3]),
            DynScalar::Binary(vec![]),
            DynScalar::Binary(vec![4, 5, 6]),
            DynScalar::Null,
        ];
        let data_type = DataType::Binary;
        let array = dynscalar_vec_to_array(values, &data_type);

        // Check the array contents
        let binary_array = array
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
            .unwrap();
        assert_eq!(binary_array.len(), 4);
        assert_eq!(binary_array.value(0), &[1, 2, 3]);
        assert_eq!(binary_array.value(1), Vec::<u8>::new());
        assert_eq!(binary_array.value(2), &[4, 5, 6]);
        assert!(binary_array.is_null(3));
    }

    #[test]
    fn test_nested_types() {
        // Test List with mixed /None values
        let values = vec![
            DynScalar::List(vec![DynScalar::Int32(1), DynScalar::Int32(2)]),
            DynScalar::List(vec![]),
            DynScalar::List(vec![
                DynScalar::Int32(3),
                DynScalar::Int32(4),
                DynScalar::Int32(5),
            ]),
            DynScalar::Null,
        ];
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let data_type = DataType::List(field);
        let array = dynscalar_vec_to_array(values, &data_type);

        // Check the array contents
        let list_array = array
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        assert_eq!(list_array.len(), 4);

        // Check first list: [1, 2]
        let first_list = list_array.value(0);
        let first_int_array = first_list
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(first_int_array.len(), 2);
        assert_eq!(first_int_array.value(0), 1);
        assert_eq!(first_int_array.value(1), 2);

        // Check second list: empty
        let value = list_array.value(1);
        let value = value.as_any().downcast_ref::<Int32Array>().unwrap();

        let expected = Int32Array::new_null(0);

        assert_eq!(value, &expected);

        // Check third list: [3, 4, 5]
        let third_list = list_array.value(2);
        let third_int_array = third_list
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        assert_eq!(third_int_array.len(), 3);
        assert_eq!(third_int_array.value(0), 3);
        assert_eq!(third_int_array.value(1), 4);
        assert_eq!(third_int_array.value(2), 5);

        // Check fourth list: null
        assert!(list_array.is_null(3));
    }
}
