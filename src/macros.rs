

#[macro_export]
macro_rules! for_all_primitivetype {
    ($macro:tt) => {
        $macro! {
            (i8,  Int8Type),
            (i16, Int16Type),
            (i32, Int32Type),
            (i64, Int64Type),
            (u8,  UInt8Type),
            (u16, UInt16Type),
            (u32, UInt32Type),
            (u64, UInt64Type),
            (f32, Float32Type),
            (f64, Float64Type)
        }
    }
}

#[macro_export]
macro_rules! for_all_primitivetype_with_variant {
    ($macro:tt) => {
        $macro! {
            (i8,  Int8Type, Int8),
            (i16, Int16Type, Int16),
            (i32, Int32Type, Int32),
            (i64, Int64Type, Int64),
            (u8,  UInt8Type, UInt8),
            (u16, UInt16Type, UInt16),
            (u32, UInt32Type, UInt32),
            (u64, UInt64Type, UInt64),
            (f32, Float32Type, Float32),
            (f64, Float64Type, Float64)
        }
    }
}


#[macro_export]
macro_rules! for_all_timetypes {
    ($macro:tt) => {
        $macro! {
            (TimestampSecond, TimestampSecondType, TimestampSecond, i64),
            (TimestampMillisecond, TimestampMillisecondType, TimestampMillisecond, i64),
            (TimestampMicrosecond, TimestampMicrosecondType, TimestampMicrosecond, i64),
            (TimestampNanosecond, TimestampNanosecondType, TimestampNanosecond, i64),
            (Time32Second, Time32SecondType, Time32Second, i32),
            (Time32Millisecond, Time32MillisecondType, Time32Millisecond, i32),
            (Time64Microsecond, Time64MicrosecondType, Time64Microsecond, i64),
            (Time64Nanosecond, Time64NanosecondType, Time64Nanosecond, i64),
            (DurationSecond, DurationSecondType, DurationSecond, i64),
            (DurationMillisecond, DurationMillisecondType, DurationMillisecond, i64),
            (DurationMicrosecond, DurationMicrosecondType, DurationMicrosecond, i64),
            (DurationNanosecond, DurationNanosecondType, DurationNanosecond, i64)
        }
    }
}

#[macro_export]
macro_rules! for_all_numerictypes {
    ($macro:tt) => {
        for_all_primitivetype!($macro);
    }
}

#[macro_export]
macro_rules! for_all_arraytypes {
    ($macro:tt) => {
        for_all_primitivetype_with_variant!($macro);
        for_all_timetypes!($macro);
        $macro! {
            (String, StringType, String, String),
            (Vec<u8>, BinaryType, Binary, Vec<u8>),
            (bool, BooleanType, Bool, bool)
        }
    }
}




