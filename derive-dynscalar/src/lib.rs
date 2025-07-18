use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

#[proc_macro_derive(IntoDynScalar)]
pub fn into_dynscalar_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Check if we're dealing with a struct
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("IntoDynScalar can only be derived for structs"),
    };

    // Generate code based on the struct fields
    let field_conversions = match fields {
        Fields::Named(fields) => {
            let field_conversions = fields.named.iter().map(|field| {
                let field_name = &field.ident;
                let field_name_str = field_name.as_ref().unwrap().to_string();
                let ty = &field.ty;

                if is_option(ty) {
                    quote! {
                        if let Some(inner) = self.#field_name {
                            map.insert(#field_name_str.to_string(), inner.into());
                        } else {
                            map.insert(#field_name_str.to_string(), DynScalar::Null);
                        }
                    }
                } else {
                    quote! {
                        map.insert(#field_name_str.to_string(), self.#field_name.into());
                    }
                }
            });

            quote! {
                let mut map = std::collections::HashMap::new();
                #(#field_conversions)*
                map
            }
        }
        Fields::Unnamed(_) => panic!("Tuple structs are not supported"),
        Fields::Unit => panic!("Unit structs are not supported"),
    };

    // Generate the implementation of From<YourType> for DynScalar
    let expanded = quote! {
        impl Into<DynScalar> for #name {
            fn into(self) -> DynScalar {
                let map = {
                    #field_conversions
                };
                DynScalar::Struct(map)
            }
        }
    };

    // Return the generated code
    TokenStream::from(expanded)
}

fn is_option(ty: &Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "Option";
        }
    }
    false
}
