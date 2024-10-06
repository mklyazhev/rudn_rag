import os


def extract_sections(txt_file):
    sections = []
    current_section = []

    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip()

            if text:
                current_section.append(text)
            elif current_section:
                sections.append("\n".join(current_section))
                current_section = []

    if current_section:
        sections.append("\n".join(current_section))

    return sections


def save_sections_to_md(sections, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, section in enumerate(sections, start=1):
        md_filename = f"{i}.md"
        md_filepath = os.path.join(output_dir, md_filename)

        with open(md_filepath, 'w', encoding='utf-8') as md_file:
            md_file.write(section)


def convert_txt_to_md(txt_file, output_dir):
    sections = extract_sections(txt_file)
    save_sections_to_md(sections, output_dir)


txt_file = r'C:\Users\gltuser\Desktop\home\git\rudn-rag\artifacts\data\lib_data\lib_data.txt'
output_dir = r'C:\Users\gltuser\Desktop\home\git\rudn-rag\artifacts\data\lib_data'
convert_txt_to_md(txt_file, output_dir)
