from setuptools import setup


install_requires = [
	"numpy",
	"frozendict",
	"frozenlist",
	"pyconll",
	"nltk",
	# needed for the tests
	"dill", # pickle package is not able to pickle the FSAs
	"pytest",
	"ply",
	"tqdm" #Helper for generating test fsas
]


setup(
	name="rayuela",
	install_requires=install_requires,
	version="0.1",
	scripts=[],
	packages=['rayuela']
)
