from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="plixkws",
    version="1.0",
    description="Plug-and-Play Multilingual Few-shot Spoken Words Recognition",
    long_description_content_type="text/markdown",
    long_description=README,
    license="Apache-2.0",
    packages=['plixkws'],
    author="Aaqib Saeed",
    author_email="aqibsaeed@protonmail.com",
    keywords=["Keyword Spotting", "Few-shot Learning", "Deep Neural Network", "Audio", "Speech"],
    url="https://github.com/FewshotML/plix",
    download_url="https://pypi.org/project/plixkws/",
)

install_requires = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "wget",
    "librosa"
]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)