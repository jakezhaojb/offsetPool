----------------------------------------------------------------------
-- progress bar
----------------------------------------------------------------------
do
   local barDone = true
   local previous = -1
   local timer
   local times
   local indices
   function progress(current, goal, str)
      if str == nil then
        str = ''  -- empty string
      end
     
      -- defaults:
      local barLength = 60 - #str
      assert(barLength > 0, 'str input is too large!')
      local smoothing = 100 
      local maxfps = 10
      
      -- Compute percentage
      local percent = math.floor(((current) * barLength) / goal)

      -- start new bar
      if (barDone and ((previous == -1) or (percent < previous))) then
         barDone = false
         previous = -1
         timer = torch.Timer()
         times = {timer:time().real}
         indices = {current}
      else
         io.write('\r')
      end

      --if (percent ~= previous and not barDone) then
      if (not barDone) then
         previous = percent
         -- print bar
         io.write(' [')
         for i=1,barLength do
            if (i < percent) then io.write('=')
            elseif (i == percent) then io.write('>')
            else io.write('.') end
         end
         io.write('] ')
         io.write(' ', current, '/', goal, ' ', str)
         -- reset for next bar
         if (percent == barLength) then
            barDone = true
            io.write('\n')
         end
         -- flush
         io.write('\r')
         io.flush()
      end
   end
end

function formatTime(seconds)
   -- decompose:
   local floor = math.floor
   local days = floor(seconds / 3600/24)
   seconds = seconds - days*3600*24
   local hours = floor(seconds / 3600)
   seconds = seconds - hours*3600
   local minutes = floor(seconds / 60)
   seconds = seconds - minutes*60
   local secondsf = floor(seconds)
   seconds = seconds - secondsf
   local millis = floor(seconds*1000)

   -- string
   local f = ''
   local i = 1
   if days > 0 then f = f .. days .. 'D' i=i+1 end
   if hours > 0 and i <= 2 then f = f .. hours .. 'h' i=i+1 end
   if minutes > 0 and i <= 2 then f = f .. minutes .. 'm' i=i+1 end
   if secondsf > 0 and i <= 2 then f = f .. secondsf .. 's' i=i+1 end
   if millis > 0 and i <= 2 then f = f .. millis .. 'ms' i=i+1 end
   if f == '' then f = '0ms' end

   -- return formatted time
   return f
end
