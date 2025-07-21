use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

#[proc_macro_derive(IntoArrowArray)]
pub fn derive_into_arrow(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("Only named struct fields supported"),
        },
        _ => panic!("Only structs supported"),
    };

    let mut builders = Vec::new();
    let mut pushers = Vec::new();
    let mut finishers = Vec::new();
    let mut names = Vec::new();
    let mut nullabilities = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let buf_ident = format_ident!("{}_data", ident);
        let nulls_ident = format_ident!("{}_validity", ident);
        let field_name = ident.to_string();
        names.push(field_name.clone());

        let ty = &field.ty;
        let is_opt = is_option(ty);
        let bare_ty = if is_opt {
            unwrap_option(ty)
        } else {
            ty.clone()
        };
        nullabilities.push(is_opt);

        let is_vec = is_vec(&bare_ty);
        let is_map = is_map(&bare_ty);

        let (elem_k, elem_v) = if is_map {
            let (k, v) = unwrap_map(&bare_ty);
            (Some(k), Some(v))
        } else {
            (None, None)
        };

        let (elem_ty, elem_is_nullable) = if is_vec {
            let inner = unwrap_vec(&bare_ty);
            if is_option(&inner) {
                (Some(unwrap_option(&inner)), true)
            } else {
                (Some(inner), false)
            }
        } else if is_map {
            (None, false)
        } else {
            (None, false)
        };

        let is_nested = match (is_vec, is_map) {
            (true, _) => !is_arrow_primitive(elem_ty.as_ref().unwrap()),
            (_, true) => true,
            _ => !is_arrow_primitive(&bare_ty),
        };

        // ====================
        // Declare
        // ====================
        let builder_decl = if is_vec {
            let elem = elem_ty.as_ref().unwrap();
            let builder_ty = arrow_builder(elem);
            let elem_datatype = arrow_datatype(elem);
            quote! {
                let mut #ident = {
                    let field = Field::new("item", #elem_datatype, #elem_is_nullable);
                    ListBuilder::new(#builder_ty::new()).with_field(field)
                };
            }
        } else if is_map || (is_nested && is_opt) {
            quote! {
                let mut #buf_ident = Vec::new();
                let mut #nulls_ident = Vec::new();
            }
        } else if is_nested {
            quote! {
                let mut #buf_ident = Vec::new();
            }
        } else {
            let builder_ty = arrow_builder(&bare_ty);
            quote! {
                let mut #ident = #builder_ty::new();
            }
        };

        // ====================
        // Push
        // ====================
        let push_code = if is_vec {
            if is_opt {
                if elem_is_nullable {
                    quote! {
                        if let Some(inner) = &item.#ident {
                            for val in inner {
                                if let Some(v) = val {
                                    #ident.values().append_value(v);
                                } else {
                                    #ident.values().append_null();
                                }
                            }
                            #ident.append(true);
                        } else {
                            #ident.append(false);
                        }
                    }
                } else {
                    quote! {
                        if let Some(inner) = &item.#ident {
                            for val in inner {
                                #ident.values().append_value(val);
                            }
                            #ident.append(true);
                        } else {
                            #ident.append(false);
                        }
                    }
                }
            } else {
                if elem_is_nullable {
                    quote! {
                        for val in &item.#ident {
                            if let Some(v) = val {
                                #ident.values().append_value(v);
                            } else {
                                #ident.values().append_null();
                            }
                        }
                        #ident.append(true);
                    }
                } else {
                    quote! {
                        for val in &item.#ident {
                            #ident.values().append_value(val);
                        }
                        #ident.append(true);
                    }
                }
            }
        } else if is_map {
            if is_opt {
                quote! {
                    if let Some(map) = &item.#ident {
                        let mut kvs = Vec::new();
                        for (k, v) in map {
                            kvs.push((k, v));
                        }
                        #buf_ident.push(kvs);
                        #nulls_ident.push(true);
                    } else {
                        #buf_ident.push(vec![]);
                        #nulls_ident.push(false);
                    }
                }
            } else {
                quote! {
                    let mut kvs = Vec::new();
                    for (k, v) in &item.#ident {
                        kvs.push((k, v));
                    }
                    #buf_ident.push(kvs);
                }
            }
        } else if is_nested {
            if is_opt {
                quote! {
                    if let Some(inner) = &item.#ident {
                        #buf_ident.push(inner.clone());
                        #nulls_ident.push(true);
                    } else {
                        #buf_ident.push(Default::default());
                        #nulls_ident.push(false);
                    }
                }
            } else {
                quote! {
                    #buf_ident.push(item.#ident.clone());
                }
            }
        } else {
            if is_opt {
                quote! {
                    if let Some(val) = &item.#ident {
                        #ident.append_value(*val);
                    } else {
                        #ident.append_null();
                    }
                }
            } else {
                quote! {
                    #ident.append_value(item.#ident.clone());
                }
            }
        };

        // ====================
        // Finish
        // ====================
        let finisher = if is_vec {
            quote! {
                Arc::new(#ident.finish()) as ArrayRef
            }
        } else if is_map {
            let k = elem_k.as_ref().unwrap();
            let v = elem_v.as_ref().unwrap();
            let inner = quote! {
                {
                    use arrow::array::*;
                    let mut offsets = Vec::with_capacity(#buf_ident.len() + 1);
                    offsets.push(0);
                    let mut keys = Vec::new();
                    let mut vals = Vec::new();
                    for kvs in &#buf_ident {
                        offsets.push(offsets.last().unwrap() + kvs.len() as i32);
                        for (k, v) in kvs {
                            keys.push(k.clone());
                            vals.push(v.clone());
                        }
                    }
                    let keys_arr = #k::into_arrow(&keys);
                    let vals_arr = #v::into_arrow(&vals);
                    let kv_struct = StructArray::from(vec![
                        ("key".to_string(), keys_arr),
                        ("value".to_string(), vals_arr),
                    ]);
                    let data = ArrayData::builder(arrow::datatypes::DataType::Map(
                            Box::new(arrow::datatypes::Field::new("entries", kv_struct.data_type().clone(), false)),
                            false))
                        .len(offsets.len() - 1)
                        .add_buffer(arrow::buffer::Buffer::from_slice_ref(&offsets))
                        .add_child_data(kv_struct.to_data())
                        .build().unwrap();
                    Arc::new(MapArray::from(data)) as ArrayRef
                }
            };
            if is_opt {
                quote! {
                    {
                        let inner = #inner;
                        let mut mask = arrow::array::builder::BooleanBufferBuilder::new(#nulls_ident.len());
                        mask.append_slice(#nulls_ident.as_slice());
                        let data = inner
                            .as_any()
                            .downcast_ref::<MapArray>()
                            .unwrap()
                            .to_data()
                            .into_builder()
                            .null_bit_buffer(Some(mask.into()))
                            .build().unwrap();
                        Arc::new(MapArray::from(data)) as ArrayRef
                    }
                }
            } else {
                inner
            }
        } else if is_nested {
            if is_opt {
                quote! {
                    {
                        let child = #bare_ty::into_arrow(&#buf_ident);
                        let mut mask = arrow::array::builder::BooleanBufferBuilder::new(#nulls_ident.len());
                        mask.append_slice(#nulls_ident.as_slice());
                        let data = child
                            .as_any()
                            .downcast_ref::<arrow::array::StructArray>()
                            .unwrap()
                            .to_data()
                            .into_builder()
                            .null_bit_buffer(Some(mask.into()))
                            .build().unwrap();
                        Arc::new(StructArray::from(data)) as ArrayRef
                    }
                }
            } else {
                quote! {
                    #bare_ty::into_arrow(&#buf_ident)
                }
            }
        } else {
            quote! {
                Arc::new(#ident.finish()) as ArrayRef
            }
        };

        builders.push(builder_decl);
        pushers.push(push_code);
        finishers.push(finisher);
    }

    let names = names.iter().map(|n| quote! { #n.to_string() });

    let nullabilities = nullabilities.iter().map(|&n| quote! { #n });

    let expanded = quote! {
        #[automatically_derived]
        impl trait_def::IntoArrowArray for #struct_name {
            fn into_arrow(vec: &[Self]) -> arrow::array::ArrayRef {
                use arrow::array::*;
                use std::sync::Arc;
                use arrow::datatypes::{DataType, Field};

                #(#builders)*

                for item in vec {
                    #(#pushers)*
                }

                let arrays = vec![#(#finishers),*];
                let fields: Vec<_> = vec![#(#names),*]
                    .into_iter()
                    .zip(arrays.iter())
                    .zip(vec![#(#nullabilities),*].into_iter())
                    .map(|((name, array), nullable)| {
                        arrow::datatypes::Field::new(name, array.data_type().clone(), nullable) // Use nullable
                    })
                    .collect();

                Arc::new(StructArray::new(fields.into(), arrays, None))
            }
        }
    };

    TokenStream::from(expanded)
}

// ====== Helpers

fn is_option(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        if p.path.segments.first().unwrap().ident == "Option" {
            return true;
        }
    }
    false
}

fn unwrap_option(ty: &Type) -> Type {
    if let Type::Path(type_path) = ty {
        let segment = &type_path.path.segments.first().unwrap();
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                return inner.clone();
            }
        }
    }
    panic!("Expected Option<T>");
}

fn is_vec(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        if p.path.segments.first().unwrap().ident == "Vec" {
            return true;
        }
    }
    false
}

fn unwrap_vec(ty: &Type) -> Type {
    if let Type::Path(type_path) = ty {
        let segment = &type_path.path.segments.first().unwrap();
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                return inner.clone();
            }
        }
    }
    panic!("Expected Vec<T>");
}

fn is_map(ty: &Type) -> bool {
    if let Type::Path(p) = ty {
        if p.path.segments.first().unwrap().ident == "HashMap" {
            return true;
        }
    }
    false
}

fn unwrap_map(ty: &Type) -> (Type, Type) {
    if let Type::Path(type_path) = ty {
        let segment = &type_path.path.segments.first().unwrap();
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            let mut iter = args.args.iter();
            let k = iter.next().unwrap();
            let v = iter.next().unwrap();
            if let (syn::GenericArgument::Type(k), syn::GenericArgument::Type(v)) = (k, v) {
                return (k.clone(), v.clone());
            }
        }
    }
    panic!("Expected HashMap<K,V>");
}

fn is_arrow_primitive(ty: &Type) -> bool {
    let s = quote!(#ty).to_string();
    matches!(
        s.as_str(),
        "String"
            | "bool"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "f32"
            | "f64"
    )
}

fn arrow_builder(ty: &Type) -> proc_macro2::TokenStream {
    match quote!(#ty).to_string().as_str() {
        "String" => quote! { StringBuilder },
        "bool" => quote! { BooleanBuilder },
        "i8" => quote! { Int8Builder },
        "i16" => quote! { Int16Builder },
        "i32" => quote! { Int32Builder },
        "i64" => quote! { Int64Builder },
        "u8" => quote! { UInt8Builder },
        "u16" => quote! { UInt16Builder },
        "u32" => quote! { UInt32Builder },
        "u64" => quote! { UInt64Builder },
        "f32" => quote! { Float32Builder },
        "f64" => quote! { Float64Builder },
        "& str" => quote! { StringBuilder },
        _ => panic!("Unsupported primitive type: {:?}", quote!(#ty).to_string()),
    }
}

fn arrow_datatype(ty: &Type) -> proc_macro2::TokenStream {
    match quote!(#ty).to_string().as_str() {
        "String" => quote! { DataType::Utf8 },
        "bool" => quote! { DataType::Boolean },
        "i8" => quote! { DataType::Int8 },
        "i16" => quote! { DataType::Int16 },
        "i32" => quote! { DataType::Int32 },
        "i64" => quote! { DataType::Int64 },
        "u8" => quote! { DataType::UInt8 },
        "u16" => quote! { DataType::UInt16 },
        "u32" => quote! { DataType::UInt32 },
        "u64" => quote! { DataType::UInt64 },
        "f32" => quote! { DataType::Float32 },
        "f64" => quote! { DataType::Float64 },
        "& str" => quote! { DataType::Utf8 },
        _ => panic!(
            "Unsupported primitive type for DataType: {:?}",
            quote!(#ty).to_string()
        ),
    }
}