use crate::array::{primitive_array::Decimal128Value, time_array::*};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum DynScalar {
    Null,

    Bool(bool),

    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),

    String(String),

    Binary(Vec<u8>),

    TimestampSecond(TimestampSecond),
    TimestampMillisecond(TimestampMillisecond),
    TimestampMicrosecond(TimestampMicrosecond),
    TimestampNanosecond(TimestampNanosecond),

    Time32Second(Time32Second),
    Time32Millisecond(Time32Millisecond),
    Time64Microsecond(Time64Microsecond),
    Time64Nanosecond(Time64Nanosecond),

    DurationSecond(DurationSecond),
    DurationMillisecond(DurationMillisecond),
    DurationMicrosecond(DurationMicrosecond),
    DurationNanosecond(DurationNanosecond),

    Decimal128(Decimal128Value),

    // Nested types
    List(Vec<DynScalar>),
    Struct(HashMap<String, DynScalar>),
    Map(Vec<(DynScalar, DynScalar)>),
    FixedSizeList(Vec<DynScalar>, i32),
}
