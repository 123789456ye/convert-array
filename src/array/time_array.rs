use crate::array::primitive_array::{ArrowTyped, TypedVec};
use crate::for_all_timetypes;
use arrow::datatypes::*;

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

#[cfg(test)]
mod tests {
    use crate::array::primitive_array::NativeArray;
    use arrow::array::{DurationSecondArray, Time32SecondArray, TimestampSecondArray};

    use super::*;

    /// Tests conversion of timestamp values to Arrow arrays.
    /// Verifies that TimestampSecond values are correctly converted to Arrow TimestampSecondArray.
    #[test]
    fn test_timestamp_types() {
        let ts_values: Vec<Option<TimestampSecond>> =
            vec![TimestampSecond(1234567890), TimestampSecond(1234567891)]
                .into_iter()
                .map(Some)
                .collect();
        let ts_vec = TimestampSecondVec::from(ts_values);
        let arrow_array = ts_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(
            arrow_array,
            TimestampSecondArray::from(vec![1234567890, 1234567891])
        );
    }

    /// Tests conversion of time values to Arrow arrays.
    /// Verifies that Time32Second values are correctly converted to Arrow Time32SecondArray.
    #[test]
    fn test_time_types() {
        let time_values: Vec<Option<Time32Second>> = vec![Time32Second(1800), Time32Second(3600)]
            .into_iter()
            .map(Some)
            .collect();
        let time_vec = Time32SecondVec::from(time_values);
        let arrow_array = time_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, Time32SecondArray::from(vec![1800, 3600]));
    }

    /// Tests conversion of duration values to Arrow arrays.
    /// Verifies that DurationSecond values are correctly converted to Arrow DurationSecondArray.
    #[test]
    fn test_duration_types() {
        let duration_values: Vec<Option<DurationSecond>> =
            vec![DurationSecond(3600), DurationSecond(7200)]
                .into_iter()
                .map(Some)
                .collect();
        let duration_vec = DurationSecondVec::from(duration_values);
        let arrow_array = duration_vec.to_arrow_array();
        assert_eq!(arrow_array.len(), 2);
        assert_eq!(arrow_array, DurationSecondArray::from(vec![3600, 7200]));
    }
}
