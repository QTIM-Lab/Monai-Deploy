-- LOCAL ADDITION, not MONAI. Stands in for the sender-side routing rule a PACS admin
-- would configure: "send new images to the AI gateway". MONAI's Informatics Gateway is
-- passive and cannot watch Orthanc, so something on the sender has to push.
--
-- Deliberately dumb. It decides nothing about which AI applies (the workflow's router
-- task does) or what MONAI accepts (the IG AE's ignoredSopClasses does). Its one rule is
-- the one only the sender can apply: don't send back what MONAI just sent us.
-- MONAI sends results from the IG's SCU AE title (configs/informatics-gateway.json,
-- dicom.scu.aeTitle), which is NOT the AE it listens on. The IG's ignoredSopClasses is
-- the second line of defence if this ever drifts.
--
-- Per series, not per study: results land in the source study, so a per-study trigger
-- would fire again on every result.

local TARGET = 'MONAI-DEPLOY'     -- IG listening AE (DicomModalities entry in orthanc.json)
local MONAI_SCU = 'MONAISCU'      -- IG export AE: results arrive from this one

function OnStableSeries(seriesId, tags, metadata)
  local series = ParseJson(RestApiGet('/series/' .. seriesId))
  local meta = ParseJson(RestApiGet('/instances/' .. series['Instances'][1] .. '/metadata?expand'))
  if meta['RemoteAET'] == MONAI_SCU then
    print('autoroute: skip series ' .. seriesId .. ' (came from ' .. MONAI_SCU .. ')')
    return
  end

  print('autoroute: send series ' .. seriesId .. ' (' .. (tags['Modality'] or '?') .. ') to ' .. TARGET)
  RestApiPost('/modalities/' .. TARGET .. '/store', DumpJson({ Resources = { seriesId }, Asynchronous = true }))
end
