import openai
import random

openai.organization = "org-WKedkVSrlLFdSiTaaMKw4bdN"
openai.api_key = "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"


def make_prompt(keywords):
    """
    input : list or str
    output : stable_diffusion에 최적화된 prompt를 거친 prompt
            
    """
    if type(keywords) == list:
        keywords = ','.join(keywords)
        prompt = "I will tell you a keyword or object, please generate 5 text prompts that would create a beautiful image, include descriptive words and language, art styles and other intricate details. Include concepts of both paintings and realistic photographs/images. Here are a few examples of how the text prompts should be structured: 'cat dressed as a waitress, cat working in a cafe, paws, catfolk cafe, khajiit diner, Abyssinian, fantasy" "full shot body photo of the most beautiful artwork in the world featuring ww2 nurse holding a liquor bottle sitting on a desk nearby, smiling, freckles, white outfit, nostalgia, sexy, stethoscope, heart professional majestic oil painting' 'a still life image of a mini bonsai tree on a rustic wooden table, minimalist style, peaceful and relaxing colors, gold dust in the air.' The keyword is {} Write all output in English (US).".format(keywords)
        return prompt
    else:
        prompt = "I will tell you a keyword or object, please generate 5 text prompts that would create a beautiful image, include descriptive words and language, art styles and other intricate details. Include concepts of both paintings and realistic photographs/images. Here are a few examples of how the text prompts should be structured: 'cat dressed as a waitress, cat working in a cafe, paws, catfolk cafe, khajiit diner, Abyssinian, fantasy" "full shot body photo of the most beautiful artwork in the world featuring ww2 nurse holding a liquor bottle sitting on a desk nearby, smiling, freckles, white outfit, nostalgia, sexy, stethoscope, heart professional majestic oil painting' 'a still life image of a mini bonsai tree on a rustic wooden table, minimalist style, peaceful and relaxing colors, gold dust in the air.' The keyword is {} Write all output in English (US).".format(keywords)
        return prompt



def make_stable_diffusion_text(prompt, samplepath):
    chatgpt_prompt = make_prompt(prompt)
    print("Prompt 응답 시작")
    response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=chatgpt_prompt,
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )

    li = response['choices'][0]['text']
    prompt_list = [sentence for sentence in str.split(li, "\n") if sentence != ""]
    with open(samplepath + "/chatgptprompt.txt", 'w', encoding="UTF-8") as f:
        for sent in prompt_list:
            f.write(sent+'\n')
    random_prompt = random.choice(prompt_list)

    return random_prompt
