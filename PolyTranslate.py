import streamlit as st
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

# Define the dictionary of language models
LANGUAGE_MODELS = {
    'Afrikaans': 'af',
    'Albanian': 'sq',
    'Amharic': 'am',
    'Arabic': 'ar',
    'Armenian': 'hy',
    'Bengali': 'bn',
    'Bosnian': 'bs',
    'Catalan': 'ca',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'Estonian': 'et',
    'Finnish': 'fi',
    'French': 'fr',
    'German': 'de',
    'Greek': 'el',
    'Gujarati': 'gu',
    'Haitian Creole': 'ht',
    'Hausa': 'ha',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Icelandic': 'is',
    'Igbo': 'ig',
    'Indonesian': 'id',
    'Irish': 'ga',
    'Italian': 'it',
    'Japanese': 'ja',
    'Kannada': 'kn',
    'Khmer': 'km',
    'Korean': 'ko',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Luxembourgish': 'lb',
    'Macedonian': 'mk',
    'Malagasy': 'mg',
    'Malayalam': 'ml',
    'Marathi': 'mr',
    'Myanmar': 'my',
    'Nepali': 'ne',
    'Norwegian': 'no',
    'Pashto': 'ps',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Punjabi': 'pa',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Scots Gaelic': 'gd',
    'Serbian': 'sr',
    'Sindhi': 'sd',
    'Sinhala': 'si',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Somali': 'so',
    'Spanish': 'es',
    'Sundanese': 'su',
    'Swahili': 'sw',
    'Swedish': 'sv',
    'Tamil': 'ta',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Vietnamese': 'vi',
    'Welsh': 'cy',
    'Xhosa': 'xh',
    'Yiddish': 'yi',
    'Yoruba': 'yo',
    'Zulu': 'zu',
}


@st.cache_resource
def load_model():
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

def translate(text, target_language, max_chunk_size=500):
    tokenizer, model = load_model()
    
    # Set the target language code for translation
    target_lang_code = LANGUAGE_MODELS.get(target_language)

    tokenizer.src_lang = "en"
    
    # Split the text into chunks if it's too long
    tokens = tokenizer.encode(text, return_tensors="pt")
    input_ids = tokens[0]
    translations = []

    for i in range(0, len(input_ids), max_chunk_size):
        chunk = input_ids[i:i + max_chunk_size].unsqueeze(0)
        generated_tokens = model.generate(**{'input_ids': chunk, 'forced_bos_token_id': tokenizer.get_lang_id(target_lang_code)})
        chunk_translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        translations.append(chunk_translation)
    
    # Combine all translated chunks
    translation = " ".join(translations)
    return translation

st.title('Welcome to PolyTranslate')
st.write('A Versatile English-to-Multilingual Translator')

text_input = st.text_area("Enter text in English:")

target_language = st.selectbox(
    'Select the target language:',
    list(LANGUAGE_MODELS.keys())
)

if st.button('Translate'):
    with st.spinner('Translating...'):
        try:
            translation = translate(text_input, target_language)
            st.write(f'Translation ({target_language}):')
            st.write(translation)
        except Exception as e:
            st.error(f"Error: {e}")
