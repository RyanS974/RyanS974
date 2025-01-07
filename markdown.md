<!-- this is a markdown comment -->
# Markdown Tutorial

This is a brief tutorial on Markdown, focusing in the GitHub style.  It is often called GitHub flavored markdown (GFM).

# Contents
- [Overview](#overview)
- [Basic Text Formatting](#basic-text-formatting)
- [Lists](#lists)
- [Task Lists](#task-lists)
- [Code](#code)
- Tables
- Links and Images
- Quotes
- HTML Tags
- Other Features

# Overview

x

# Basic Text Formatting

Basic text formatting is done mainly with asterisks.

This code:

`*this is italic text*`

Produces this:

*this is italic text*

This code:

`**this is bold text**`

Produces this:

**this is bold text**

This code:

`~~this is strikethrough~~`

Produces this:

~~this is strikethrough~~

Here are some combinations you can use also:

This code:

`***this is both bold and italic***`

Produces this:

***this is both bold and italic***

<u>Summary:</u>

One asterisk produces italic.
Two asterisks produce bold.
Two tildes produce strikethrough.
Three asterisks produce the bold and italic combination.

In the HTML Tags section, we will go over using a subset of HTML that works with GitHub markdown, such as underscore being implemented with `<u>underscored text</u>`.

# Lists

Lists have several prefixes that can modify the list.  They are generally referred to as unordered lists and ordered lists.  The ordered ones are numbered.

<u>Key Terms:</u>

- Unordered List
  - These are lists with a symbol such as a dash, plus, or asterisk.
- Ordered List
  - Also called numbered lists. These are number formatting.

## Unordered Lists

The -, +, and * characters make a standard unordered list.

This code:

```
- dash list
  - level two
```

Produces this:

- dash list
  - level two

NOTE: You need to use a tab or at least two spaces to indent the list for the second level.

This code:

```
+ plus list
  + level two
```

Produces this:

+ plus list
  + level two

This code:

```
* asterisk list
  * level two
```

Produces this:

* asterisk list
  * level two

## Ordered (Numbered) Lists

This code:

```
1. Item 1
   1. Item 1.1
2. Item 2
   1. Item 2.1
      1. Item 2.2
```

Produces this:

1. Item 1
   1. Item 1.1
2. Item 2
   1. Item 2.1
      1. Item 2.2

NOTE: You need three spaces or a tab for the indentation to work with numbered lists.  As noted above, you only need two spaces or a tab for the unordered lists.

Also, as you can see, the numbering is not what might be expected.  For the numbers other than one, you still have to use `1.` for the indented levels.  There appears to be no other method for this at the moment with GitHub markdown.

# Task Lists

You can make task lists with GFM, GitHub Flavor Markdown.

This code:

```
- [ ] task 1
- [x] task 2
```

Produces this:

- [ ] task 1
- [x] task 2

Put an x in the box when a task is completed.

# Code

You can format code with a code block or inline.

## Code Block

This code:

back

````
```
This is a code block
```
````

Produces this:

```
This is a code block
```

NOTE: Four backticks are required to have three backticks appear.  The code above is actually four backticks enclosing the other three.  If you use three backticks to enclose your code block, you can include one or two backticks and it will appear.

```
`
``
```

## Inline Code

Inline code can be enclosed with a single backtick, such as `this is code`.

This code:

```
This is an inline `code block`.
```

Produces this:

This is an inline `code block`.

Markdown itself does not natively support syntax highlighting for different programming languages. However, GitHub Flavored Markdown (GFM) and many other Markdown processors (like those used in static site generators, IDEs, and other platforms) extend the basic Markdown syntax to support syntax highlighting for a wide range of programming languages.

Here are some programming languages supported by GitHub and other markdown processors:

- Python: ```python
- JavaScript: ```javascript
- Java: ```java
- C++: ```cpp
- Ruby: ```ruby
- JSON: ```json
- HTML: ```html
- CSS: ```css
- Shell: ```sh
- SQL: ```sql


# Tables

To create tables in GitHub Flavored Markdown (GFM), you can use pipes (|) to separate columns and hyphens (-) to create the header row.

This code:

```
| column 1 | column 2 | column 3 |
|---------:|:--------:|:---------|
| Row 1    | Data     | Data     |
| Row 2    | Data     | Data     |
```

Produces this:

| column 1 | column 2 | column 3 |
|---------:|:--------:|:---------|
| Row 1    | Data     | Data     |
| Row 2    | Data     | Data     |

You can align text within columns by adding colons (:) to the header row:

* :---  for left alignment  
* :---: for center alignment  
*  ---: for right alignment

GitHub Flavored Markdown (GFM) does not support table cell borders directly. Markdown tables are rendered without borders. If you need to add borders to your tables, you would need to use HTML within your Markdown file.

# Links and Images

You can have links and images in markdown.  Links can be URLs or even anchor based inside of a markdown file.

## Links

This code:

`[HTML Link text](http://www.github.com)`

Produces this:

[HTML Link text](http:/www.github.com)

You can also link inside of a markdown file.

This code:

`[Link to the Table of Contents](#contents)`

Produces this:

[Link to the Table of Contents](#contents)

Any heading is a possible anchor, such as the one above `#contents`.  If there is a space, such as with `# Task Lists`, there is a hyphen inserted, so it would be `#task-lists`.

You can also make your own anchors without headings using html div tags with the id attribute.

This code:

`<div id="custom-section"></div>`

<div id="custom-section"></div>

... will insert an invisible tag.  There isn't supposed to be anything seen on this directly.  We will link to it in code.

This code:

`[Go to the custom-section div tag id](#custom-section)`

Produces this:

[Go to the custom-section div tag id](#custom-section)

Here is an in-file link to the Table of Contents at the top of this document, which also contains links to the level one headings of this document.

[Go to the Table of Contents](#contents)

## Images

Images syntax is similar to links, but adds an \!.

This code:

```
![Alt text for the image](https://picsum.photos/256/256)
```

Produces this:

![Alt text for the image](https://picsum.photos/256/256)

Also, you can link to an image inside your local directory structure.

This code:

```
![An image in this directory](./image.png)
```

Produces this:

![An image in this directory](./image.png)

# Blockquotes

You can use blockquotes to emphasis text.

This code:

```
> blockquote line 1.  
> blockquote line 2.
> blockquote line 3.
```

Produces this:

> blockquote line 1.    
> blockquote line 2.  
> blockquote line 3.

This code:  

```
> blockquote line 1. (only this line here uses a >)  
blockquote line 2.
blockquote line 3.
```

Produces this:

> blockquote line 1. (only this line here uses a >)  
blockquote line 2.  
blockquote line 3.

Remember to use at least two spaces at the end of a line to create a new line. Also, the second block of blockquotes just uses the greater than sign, \>, once for the first line.  As long as you don't have an empty line after that first line, it will continue with the blockquote format.  This text paragraph after the previous second blockquote lines has a blank line, so this paragraph is not included in that blockquote.

# HTML Tags

A subset of html tags are supported by some markdown interpreters, such as GitHub.

- underline, `<u>underlined text</u>`
  <u>underlined text</u>
- break, `<br>`, this produces a line break.
- superscript, `Text<sup>superscript text</sup>`
  - Text<sup>superscript text</sup>
- subscript, `<sub>subscript text</sub>`


# Other Features

There are various other features of markdown covered below.

## Escape Sequences

You can escape most special characters with a backslash.

This code:

```
\* test (it won't produce a bullet point)  
\+ test (it won't produce a bullet point)  
\- test (it won't produce a bullet point)
```

Produces this:

\* test (it won't produce a bullet point)  
\+ test (it won't produce a bullet point)  
\- test (it won't produce a bullet point)


## New Lines

If you end a line with two or more spaces it will create a new line.

This code:

```
Here is text on a line.
Here is another line after a return, but no spaces at the end of the above line.
```

Produces this:

Here is text on a line.
Here is another line after a return, but no spaces at the end of the above line.

This code:

```
Here is text on a line.  
Here is another line after a return, and with two spaces at the end.
```

Produces this:

Here is text on a line.  
Here is another line after a return, and with two spaces at the end.

