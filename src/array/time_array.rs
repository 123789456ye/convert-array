use crate::array::primitive_array::{ArrowTyped, TypedVec};
use arrow::{datatypes::*};
use crate::for_all_timetypes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampSecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampMillisecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampMicrosecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampNanosecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time32Second(pub i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time32Millisecond(pub i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time64Microsecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Time64Nanosecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationSecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationMillisecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationMicrosecond(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationNanosecond(pub i64);



/// Implements ArrowTyped trait and From conversions for time-related types.
/// This macro generates trait implementations for timestamp, time, and duration types,
/// enabling conversion between custom wrapper types and their underlying primitive values.
macro_rules! impl_arrowtyped_for_time_types {
    ($(($newtype:ty, $arrow_ty:ty, $variant:ident, $inner:ty)),*) => {
        $(
            impl ArrowTyped for $newtype {
                type ArrowType = $arrow_ty;
            }
            
            impl From<$newtype> for $inner {
                fn from(val: $newtype) -> Self {
                    val.0
                }
            }
            
            impl From<$inner> for $newtype {
                fn from(val: $inner) -> Self {
                    Self(val)
                }
            }
        )*
    }
}

for_all_timetypes!(impl_arrowtyped_for_time_types);

pub type TimestampSecondVec = TypedVec<TimestampSecondType>;
pub type TimestampMillisecondVec = TypedVec<TimestampMillisecondType>;
pub type TimestampMicrosecondVec = TypedVec<TimestampMicrosecondType>;
pub type TimestampNanosecondVec = TypedVec<TimestampNanosecondType>;
pub type Time32SecondVec = TypedVec<Time32SecondType>;
pub type Time32MillisecondVec = TypedVec<Time32MillisecondType>;
pub type Time64MicrosecondVec = TypedVec<Time64MicrosecondType>;
pub type Time64NanosecondVec = TypedVec<Time64NanosecondType>;
pub type DurationSecondVec = TypedVec<DurationSecondType>;
pub type DurationMillisecondVec = TypedVec<DurationMillisecondType>;
pub type DurationMicrosecondVec = TypedVec<DurationMicrosecondType>;
pub type DurationNanosecondVec = TypedVec<DurationNanosecondType>;

/// Type aliases for optional time types using TypedOptVec
pub type TimestampSecondOptVec = crate::array::primitive_array::TypedOptVec<TimestampSecondType>;
pub type TimestampMillisecondOptVec = crate::array::primitive_array::TypedOptVec<TimestampMillisecondType>;
pub type TimestampMicrosecondOptVec = crate::array::primitive_array::TypedOptVec<TimestampMicrosecondType>;
pub type TimestampNanosecondOptVec = crate::array::primitive_array::TypedOptVec<TimestampNanosecondType>;
pub type Time32SecondOptVec = crate::array::primitive_array::TypedOptVec<Time32SecondType>;
pub type Time32MillisecondOptVec = crate::array::primitive_array::TypedOptVec<Time32MillisecondType>;
pub type Time64MicrosecondOptVec = crate::array::primitive_array::TypedOptVec<Time64MicrosecondType>;
pub type Time64NanosecondOptVec = crate::array::primitive_array::TypedOptVec<Time64NanosecondType>;
pub type DurationSecondOptVec = crate::array::primitive_array::TypedOptVec<DurationSecondType>;
pub type DurationMillisecondOptVec = crate::array::primitive_array::TypedOptVec<DurationMillisecondType>;
pub type DurationMicrosecondOptVec = crate::array::primitive_array::TypedOptVec<DurationMicrosecondType>;
pub type DurationNanosecondOptVec = crate::array::primitive_array::TypedOptVec<DurationNanosecondType>;

#[cfg(test)]
mod tests {
    use arrow::array::{DurationSecondArray, Time32SecondArray, TimestampSecondArray};
    use crate::array::primitive_array::NativeArray;

    use super::*;

    /// Tests conversion of timestamp values to Arrow arrays.
    /// Verifies that TimestampSecond values are correctly converted to Arrow TimestampSecondArray.
    #[test]
    fn test_timestamp_types() {
        let ts_values = vec![
            TimestampSecond(1234567890),
            TimestampSecond(1234567891),
        ];
        let ts_vec = TimestampSecondVec::from(ts_values);
        let arrow_array = ts_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, TimestampSecondArray::from(vec![1234567890, 1234567891]));
    }

    /// Tests conversion of time values to Arrow arrays.
    /// Verifies that Time32Second values are correctly converted to Arrow Time32SecondArray.
    #[test]
    fn test_time_types() {
        let time_values = vec![
            Time32Second(1800),
            Time32Second(3600),
        ];
        let time_vec = Time32SecondVec::from(time_values);
        let arrow_array = time_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, Time32SecondArray::from(vec![1800, 3600]));
    }

    /// Tests conversion of duration values to Arrow arrays.
    /// Verifies that DurationSecond values are correctly converted to Arrow DurationSecondArray.
    #[test]
    fn test_duration_types() {
        let duration_values = vec![
            DurationSecond(3600),
            DurationSecond(7200),
        ];
        let duration_vec = DurationSecondVec::from(duration_values);
        let arrow_array = duration_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, DurationSecondArray::from(vec![3600, 7200]));
    }
}