---
layout: book
title: "CSS Styling"
book: "web-dev"
type: "chapter"
order: 2
---

# CSS Styling

## Basic CSS Syntax

CSS uses selectors to style HTML elements:

```css
/* Element selector */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}

/* Class selector */
.highlight {
    background-color: yellow;
    padding: 5px;
}

/* ID selector */
#header {
    background-color: #333;
    color: white;
}
```

## Box Model

The CSS box model consists of:
- Content
- Padding
- Border
