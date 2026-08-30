--- Promote the report's opening headings into Pandoc title metadata.

function Pandoc(document)
  local title = document.blocks[1]
  local subtitle = document.blocks[2]
  if title and title.t == "Header" and title.level == 1 then
    document.meta.title = pandoc.MetaInlines(title.content)
    document.blocks:remove(1)
  end
  if subtitle and subtitle.t == "Header" and subtitle.level == 2 then
    document.meta.subtitle = pandoc.MetaInlines(subtitle.content)
    document.blocks:remove(1)
  end
  return document
end
