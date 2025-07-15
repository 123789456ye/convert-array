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
    
    // Optional variants for all types
    OptionalBool(Option<bool>),
    
    OptionalInt8(Option<i8>),
    OptionalInt16(Option<i16>),
    OptionalInt32(Option<i32>),
    OptionalInt64(Option<i64>),
    OptionalUInt8(Option<u8>),
    OptionalUInt16(Option<u16>),
    OptionalUInt32(Option<u32>),
    OptionalUInt64(Option<u64>),
    OptionalFloat32(Option<f32>),
    OptionalFloat64(Option<f64>),
    
    OptionalString(Option<String>),
    OptionalBinary(Option<Vec<u8>>),
    
    OptionalTimestampSecond(Option<TimestampSecond>),
    OptionalTimestampMillisecond(Option<TimestampMillisecond>),
    OptionalTimestampMicrosecond(Option<TimestampMicrosecond>),
    OptionalTimestampNanosecond(Option<TimestampNanosecond>),
    
    OptionalTime32Second(Option<Time32Second>),
    OptionalTime32Millisecond(Option<Time32Millisecond>),
    OptionalTime64Microsecond(Option<Time64Microsecond>),
    OptionalTime64Nanosecond(Option<Time64Nanosecond>),
    
    OptionalDurationSecond(Option<DurationSecond>),
    OptionalDurationMillisecond(Option<DurationMillisecond>),
    OptionalDurationMicrosecond(Option<DurationMicrosecond>),
    OptionalDurationNanosecond(Option<DurationNanosecond>),
    
    OptionalDecimal128(Option<Decimal128Value>),
    
    // Optional nested types
    OptionalList(Option<Vec<DynScalar>>),
    OptionalStruct(Option<HashMap<String, DynScalar>>),
    OptionalMap(Option<Vec<(DynScalar, DynScalar)>>),
    OptionalFixedSizeList(Option<Vec<DynScalar>>, i32),
}



