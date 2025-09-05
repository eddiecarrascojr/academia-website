export const profile = {
	fullName: 'Eddie Carrasco Jr',
	title: 'Machine Learning Engineer',
	institute: 'Microsoft',
	author_name: 'Eduardo Carrasco Jr.', // Author name to be highlighted in the papers section
	research_areas: [
			{ title: 'Evolutionary AI', description: 'Swarm and Evolutionary Intelligence', field: 'ai' },
			{ title: 'Complex Systems', description: 'Emergent behavior in complex systems', field: 'complex-systems' },
			{ title: 'Large Language Models', description: 'Behavior and applications of large language models', field: 'llms' },
			{ title: 'Self Reinforcement Learning', description: 'Focus on improving reinforcement learning system', field: 'deep-rl' },
	],
}

// Set equal to an empty string to hide the icon that you don't want to display
export const social = {
	youtube: 'https://www.youtube.com/@EddieInTech/',
	linkedin: 'https://www.linkedin.com/in/eddiecarrascojr/',
	instagram: 'https://www.instagram.com/eddiecarrascojr.ai/',
	github: 'https://github.com/eddiecarrascojr',
	scholar: 'https://scholar.google.com/citations?user=Kq3A83AAAAAJ&hl=en',
	arxiv: '',
}

export const template = {
	website_url: 'https://localhost:4321', // Astro needs to know your site’s deployed URL to generate a sitemap. It must start with http:// or https://
	menu_left: false,
	transitions: true,
	darkTheme: 'dark', // Select one of the Daisy UI Themes or create your own
	excerptLength: 200,
	postPerPage: 3,
    base: '' // Repository name starting with /
}

export const seo = {
	default_title: 'Astro Academia',
	default_description: 'Exploring AI Research through experimental approaches and education.',
	default_image: '/images/astro-academia.png',
}
